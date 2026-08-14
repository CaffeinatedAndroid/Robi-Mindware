import numpy as np
import cv2
import os
import time
import psutil
import queue
import sounddevice as sd

try:
    from scipy.io import wavfile
except ImportError:
    wavfile = None
    print("[WARNING] scipy not found — WAV saving will be disabled.")
# ============ WAVEFORM CONFIG ============
SAMPLE_RATE     = 16000
CHUNK_SECONDS   = 1.0
CHUNK_SAMPLES   = int(SAMPLE_RATE * CHUNK_SECONDS)
IN_CHANNELS     = 1

# ============ SNN CONFIG ============
NUM_FILTERS     = 32         # Was 24 — 120 encoder weights can't encode 16k samples
KERNEL_SIZE     = 9          # Was 5 — larger temporal receptive field
PADDING         = (KERNEL_SIZE - 1) // 2
WAV_POOL_SIZE   = 10         # 16000 / 25 = 640 (exact)
assert CHUNK_SAMPLES % WAV_POOL_SIZE == 0, \
    f"CHUNK_SAMPLES ({CHUNK_SAMPLES}) must be divisible by WAV_POOL_SIZE ({WAV_POOL_SIZE})"

LEARNING_RATE   = 0.0001     # Was 0.00007 — too low for waveform scale
EPOCHS          = 30
CAPTURE_SECONDS = 30
DATASET_DIR     = "waveform_dataset"
WEIGHTS_FILE    = "waveform_snn_weights.bin"
TRAIN_OUT_DIR   = "waveform_snn_training_output"
POOL_TYPE       = "avg"

T_STEPS         = 10         # Was 30 — lower variance, faster epochs
DT              = 1.0
TAU_MEM         = 20.0
TAU_SYN         = 5.0
V_TH            = 0.6       # Was 0.1 — CRITICAL FIX. 0.1 = all neurons fire constantly = no information
V_RESET         = 0.0
MAX_INPUT_RATE  = 1.0        # Full rate for signed spikes
SURROGATE_BETA  = 1.0        # Was 0.2 — with V_TH=1.0, sharper grad is fine

WEIGHT_CLIP     = 2.0
WEIGHT_DECAY    = 0.0        # Keep disabled — kills tiny nets on huge data
SPARSITY_LAMBDA = 1e-5
TARGET_RATE     = 0.15       # ~2 spikes/neuron per 20-step window
HOMEOSTATIC_LR  = 0.0001      # Was 0.005 — gentler
USE_DALES       = False
GRAD_CLIP       = 10.0       # Was 4.0 — waveform MSE over 16k samples needs headroom
# =======================================



# ---------- 1D Convolution ----------
def conv1d(signal, kernel, padding=0):
    if padding > 0:
        signal = np.pad(signal, (padding, padding), mode='constant', constant_values=0)
    L = len(signal)
    K = len(kernel)
    out_len = L - K + 1
    if out_len <= 0:
        return np.zeros(1, dtype=np.float32)
    shape = (out_len, K)
    strides = (signal.strides[0], signal.strides[0])
    patches = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)
    return np.sum(patches * kernel, axis=1)


def conv1d_grad_kernel(signal, grad_output, k_shape):
    K = k_shape[0]
    pad = (K - 1) // 2
    padded = np.pad(signal, (pad, pad), mode='constant', constant_values=0)
    shape = (len(grad_output), K)
    strides = (padded.strides[0], padded.strides[0])
    patches = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return np.sum(patches * grad_output[:, None], axis=0)


def conv1d_grad_input(grad_output, kernel, target_len):
    rot_k = np.flip(kernel)
    pad = (len(kernel) - 1) // 2
    padded = np.pad(grad_output, (pad, pad), mode='constant', constant_values=0)
    result = conv1d(padded, rot_k, padding=0)
    if len(result) != target_len:
        if len(result) > target_len:
            start = (len(result) - target_len) // 2
            result = result[start:start + target_len]
        else:
            result = np.pad(result, (0, target_len - len(result)), mode='constant')
    return result


# ---------- 1D Pooling ----------
def maxpool1d_forward(x, pool_size=2):
    L = len(x)
    L -= L % pool_size
    x = x[:L]
    out_len = L // pool_size
    pooled = np.zeros(out_len, dtype=np.float32)
    mask = np.zeros(L, dtype=np.float32)
    for i in range(out_len):
        patch = x[i * pool_size:(i + 1) * pool_size]
        pooled[i] = np.max(patch)
        idx = np.argmax(patch)
        mask[i * pool_size + idx] = 1.0
    return pooled, mask


def maxpool1d_backward(d_pooled, mask, pool_size=2):
    L = len(mask)
    dx = np.zeros(L, dtype=np.float32)
    for i in range(len(d_pooled)):
        dx[i * pool_size:(i + 1) * pool_size] = mask[i * pool_size:(i + 1) * pool_size] * d_pooled[i]
    return dx


def avgpool1d_forward(x, pool_size=2):
    L = len(x)
    L -= L % pool_size
    x = x[:L]
    out_len = L // pool_size
    pooled = np.zeros(out_len, dtype=np.float32)
    for i in range(out_len):
        pooled[i] = np.mean(x[i * pool_size:(i + 1) * pool_size])
    return pooled


def avgpool1d_backward(d_pooled, target_len, pool_size=2):
    dx = np.zeros(target_len, dtype=np.float32)
    for i in range(len(d_pooled)):
        dx[i * pool_size:(i + 1) * pool_size] = d_pooled[i] * (1.0 / pool_size)
    return dx


def upsample1d(x, target_len, pool_size=2):
    out = np.zeros(target_len, dtype=np.float32)
    for i in range(len(x)):
        out[i * pool_size:(i + 1) * pool_size] = x[i]
    return out


# ---------- Audio Utilities ----------
def normalize_audio(audio, target_peak=0.95):
    peak = np.max(np.abs(audio))
    if peak > 1e-8:
        return np.clip(audio / peak * target_peak, -1.0, 1.0)
    return audio


def audio_to_int16(audio):
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)


# ---------- Visualization ----------
def draw_waveform(waveform, height=300, width=1200, color=(0, 255, 0)):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    L = len(waveform)
    step = max(1, L // width)
    w_disp = waveform[::step][:width]

    def map_y(v):
        y = int(height // 2 - v * (height // 2 - 10))
        return np.clip(y, 0, height - 1)

    for i in range(len(w_disp) - 1):
        x1, x2 = i, i + 1
        y1 = map_y(w_disp[i])
        y2 = map_y(w_disp[i + 1])
        cv2.line(img, (x1, y1), (x2, y2), color, 1)
    return img


# ---------- SNN Model ----------
class WaveformSNNAutoencoder:
    def __init__(self, in_channels=IN_CHANNELS):
        self.in_channels = in_channels
        fan_in = KERNEL_SIZE * in_channels

        self.W_enc = np.random.randn(NUM_FILTERS, KERNEL_SIZE, in_channels).astype(np.float32)
        self.W_enc *= np.sqrt(2.0 / fan_in) * 0.5

        if USE_DALES and NUM_FILTERS % 2 == 0:
            self.E_mask = np.ones((NUM_FILTERS, 1, 1), dtype=np.float32)
            self.I_mask = np.ones_like(self.E_mask)
            self.E_mask[NUM_FILTERS // 2:] = 0
            self.I_mask[:NUM_FILTERS // 2] = 0
            self.W_enc = np.abs(self.W_enc) * self.E_mask - np.abs(self.W_enc) * self.I_mask

        self.W_dec = np.random.randn(NUM_FILTERS, KERNEL_SIZE, in_channels).astype(np.float32) * 0.05

        self.b_enc = np.zeros(NUM_FILTERS, dtype=np.float32)
        self.b_dec = np.full(in_channels, 0.0, dtype=np.float32)

        self.alpha = np.exp(-DT / TAU_MEM)
        self.beta = np.exp(-DT / TAU_SYN)
        self.v_th = np.full(NUM_FILTERS, V_TH, dtype=np.float32)

        self.x = None
        self.spike_prob = None
        self.expected_input = None
        self.rate_enc = None
        self.V_pre_trace = None
        self.pooled = None
        self.pool_masks = None
        self.upsampled = None
        self.output = None

    def _clip_weights(self):
        np.clip(self.W_enc, -WEIGHT_CLIP, WEIGHT_CLIP, out=self.W_enc)
        np.clip(self.W_dec, -WEIGHT_CLIP, WEIGHT_CLIP, out=self.W_dec)

    def _enforce_dales(self):
        if USE_DALES and NUM_FILTERS % 2 == 0:
            self.W_enc = np.abs(self.W_enc) * self.E_mask - np.abs(self.W_enc) * self.I_mask

    def _clip_grad(self, grad):
        return np.clip(grad, -GRAD_CLIP, GRAD_CLIP)

    def _check_nan(self, name):
        for tensor, label in [(self.W_enc, "W_enc"), (self.W_dec, "W_dec"),
                              (self.b_enc, "b_enc"), (self.b_dec, "b_dec"),
                              (self.v_th, "v_th")]:
            if not np.isfinite(tensor).all():
                print(f"  [WARNING] NaN/Inf detected in {label} after {name}!")
                return True
        return False

    def surrogate_grad(self, V_pre):
        x = V_pre - self.v_th[:, None]
        return 1.0 / (1.0 + np.abs(SURROGATE_BETA * x))**2

    def forward(self, x):
        self.x = x.astype(np.float32)
        L, C = self.x.shape
        assert C == self.in_channels

        # === SIGNED POISSON ENCODING ===
        # Amplitude -> spike probability, sign -> spike polarity
        self.spike_prob = np.clip(np.abs(self.x) * MAX_INPUT_RATE, 0.0, 1.0)
        self.expected_input = self.spike_prob * np.sign(self.x)

        V = np.zeros((NUM_FILTERS, L), dtype=np.float32)
        I_syn = np.zeros((NUM_FILTERS, L), dtype=np.float32)
        S_accum = np.zeros((NUM_FILTERS, L), dtype=np.float32)
        V_pre_trace = []

        for t in range(T_STEPS):
            rand_mask = np.random.rand(L, C)
            S_in = np.where(rand_mask < self.spike_prob, np.sign(self.x), 0.0).astype(np.float32)

            I_t = np.zeros((NUM_FILTERS, L), dtype=np.float32)
            for f in range(NUM_FILTERS):
                for c in range(C):
                    I_t[f] += conv1d(S_in[:, c], self.W_enc[f, :, c], padding=PADDING)
                I_t[f] += self.b_enc[f]

            I_syn = self.beta * I_syn + I_t
            V_pre = self.alpha * V + I_syn
            spike = (V_pre >= self.v_th[:, None]).astype(np.float32)
            V = np.where(spike > 0, V_RESET, V_pre)

            S_accum += spike
            V_pre_trace.append(V_pre.copy())

        self.rate_enc = S_accum / T_STEPS
        self.V_pre_trace = np.array(V_pre_trace)

        # Pool
        pooled_len = L // WAV_POOL_SIZE
        self.pooled = np.zeros((NUM_FILTERS, pooled_len), dtype=np.float32)
        self.pool_masks = []
        for f in range(NUM_FILTERS):
            if POOL_TYPE == "max":
                p, m = maxpool1d_forward(self.rate_enc[f], pool_size=WAV_POOL_SIZE)
                self.pooled[f] = p
                self.pool_masks.append(m)
            else:
                self.pooled[f] = avgpool1d_forward(self.rate_enc[f], pool_size=WAV_POOL_SIZE)
                self.pool_masks.append(None)

        # Upsample
        self.upsampled = np.zeros((NUM_FILTERS, L), dtype=np.float32)
        for f in range(NUM_FILTERS):
            self.upsampled[f] = upsample1d(self.pooled[f], target_len=L, pool_size=WAV_POOL_SIZE)

        # Decode
        self.output = np.zeros((L, C), dtype=np.float32)
        for f in range(NUM_FILTERS):
            for c in range(C):
                self.output[:, c] += conv1d(self.upsampled[f], self.W_dec[f, :, c], padding=PADDING)
        self.output += self.b_dec
        self.output = np.clip(self.output, -2.0, 2.0)

        return self.output

    def backward(self, target):
        C = self.in_channels
        L = self.x.shape[0]
        d_pre = self.output - target

        self.b_dec -= LEARNING_RATE * np.sum(d_pre, axis=0)
        np.clip(self.b_dec, -2.0, 2.0, out=self.b_dec)

        dW_dec = np.zeros_like(self.W_dec)
        d_upsampled = np.zeros_like(self.upsampled)
        for f in range(NUM_FILTERS):
            for c in range(C):
                dW_dec[f, :, c] += conv1d_grad_kernel(
                    self.upsampled[f], d_pre[:, c], (KERNEL_SIZE,)
                )
                d_upsampled[f] += conv1d_grad_input(
                    d_pre[:, c], self.W_dec[f, :, c], L
                )

        dW_dec = self._clip_grad(dW_dec)
        self.W_dec -= LEARNING_RATE * dW_dec

        # Upsample backprop
        d_pooled = np.zeros_like(self.pooled)
        for f in range(NUM_FILTERS):
            for i in range(self.pooled.shape[1]):
                d_pooled[f, i] = np.sum(d_upsampled[f, i * WAV_POOL_SIZE:(i + 1) * WAV_POOL_SIZE])

        # Pool backprop
        d_rate_enc = np.zeros_like(self.rate_enc)
        for f in range(NUM_FILTERS):
            if POOL_TYPE == "max":
                d_rate_enc[f] = maxpool1d_backward(d_pooled[f], self.pool_masks[f], pool_size=WAV_POOL_SIZE)
            else:
                d_rate_enc[f] = avgpool1d_backward(d_pooled[f], L, pool_size=WAV_POOL_SIZE)

        d_rate_enc += SPARSITY_LAMBDA * np.sign(self.rate_enc)

        for f in range(NUM_FILTERS):
            mean_rate = np.mean(self.rate_enc[f])
            error = mean_rate - TARGET_RATE
            self.b_enc[f] -= HOMEOSTATIC_LR * error
        np.clip(self.b_enc, -2.0, 2.0, out=self.b_enc)

        for f in range(NUM_FILTERS):
            mean_rate = np.mean(self.rate_enc[f])
            self.v_th[f] += 0.01 * (mean_rate - TARGET_RATE)
        self.v_th = np.clip(self.v_th, 0.5, 2.0)

        sg = np.zeros_like(self.rate_enc)
        for t in range(T_STEPS):
            sg += self.surrogate_grad(self.V_pre_trace[t])
        sg /= T_STEPS

        dI_avg = d_rate_enc * sg

        # === ENCODER GRADIENT USES SIGNED EXPECTED INPUT ===
        dW_enc = np.zeros_like(self.W_enc)
        for f in range(NUM_FILTERS):
            for c in range(C):
                dW_enc[f, :, c] += conv1d_grad_kernel(
                    self.expected_input[:, c], dI_avg[f], (KERNEL_SIZE,)
                )

        dW_enc = self._clip_grad(dW_enc)
        self.W_enc -= LEARNING_RATE * dW_enc

        self.W_enc -= LEARNING_RATE * WEIGHT_DECAY * self.W_enc
        self.W_dec -= LEARNING_RATE * WEIGHT_DECAY * self.W_dec

        self._clip_weights()
        self._enforce_dales()
        self._check_nan("backward")

    def train_step(self, x):
        out = self.forward(x)
        loss = 0.5 * np.sum((out - x) ** 2)
        self.backward(x)
        return loss

    def get_firing_stats(self):
        mean_rate = np.mean(self.rate_enc)
        max_rate = np.max(self.rate_enc)
        return mean_rate, max_rate

    def save_weights(self, path):
        flat = np.concatenate([
            self.W_enc.flatten(),
            self.W_dec.flatten(),
            self.b_enc.flatten(),
            self.b_dec.flatten(),
            self.v_th.flatten()
        ]).astype(np.float32)
        flat.tofile(path)
        print(f"Saved {len(flat)} floats -> {path}")

    def load_weights(self, path):
        flat = np.fromfile(path, dtype=np.float32)
        idx = 0
        n_enc = NUM_FILTERS * KERNEL_SIZE * self.in_channels
        n_dec = NUM_FILTERS * KERNEL_SIZE * self.in_channels
        self.W_enc = flat[idx:idx + n_enc].reshape(NUM_FILTERS, KERNEL_SIZE, self.in_channels)
        idx += n_enc
        self.W_dec = flat[idx:idx + n_dec].reshape(NUM_FILTERS, KERNEL_SIZE, self.in_channels)
        idx += n_dec
        self.b_enc = flat[idx:idx + NUM_FILTERS]
        idx += NUM_FILTERS
        self.b_dec = flat[idx:idx + self.in_channels]
        idx += self.in_channels
        self.v_th = flat[idx:idx + NUM_FILTERS]
        print(f"Loaded weights from {path}")


# ---------- Dataset ----------
def capture_waveform_dataset():
    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"Recording {CAPTURE_SECONDS}s of raw waveform audio...")
    total_samples = int(CAPTURE_SECONDS * SAMPLE_RATE)
    recording = sd.rec(total_samples, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    recording = recording.flatten()

    n_chunks = len(recording) // CHUNK_SAMPLES
    chunks = []
    for i in range(n_chunks):
        chunk = recording[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]
        chunk = chunk.reshape(CHUNK_SAMPLES, 1)
        np.save(os.path.join(DATASET_DIR, f"wave_{i:04d}.npy"), chunk)
        chunks.append(chunk)

    print(f"Saved {n_chunks} waveform chunks to ./{DATASET_DIR}/")
    return chunks


def load_waveform_dataset():
    if not os.path.exists(DATASET_DIR):
        print(f"Directory '{DATASET_DIR}' does not exist.")
        return []
    files = sorted(f for f in os.listdir(DATASET_DIR) if f.endswith('.npy'))
    if not files:
        print(f"No .npy files found in '{DATASET_DIR}'.")
    chunks = []
    for f in files:
        chunk = np.load(os.path.join(DATASET_DIR, f))
        if chunk.shape[0] >= CHUNK_SAMPLES:
            chunks.append(chunk[:CHUNK_SAMPLES].reshape(CHUNK_SAMPLES, 1))
        else:
            print(f"  Skipping {f}: too short ({chunk.shape[0]} < {CHUNK_SAMPLES})")
    print(f"Loaded {len(chunks)} waveforms from ./{DATASET_DIR}/")
    return chunks


# ---------- Training & Demo ----------
def save_training_outputs(model, test_wave, epoch, folder):
    os.makedirs(folder, exist_ok=True)
    recon = model.forward(test_wave)

    orig_img = draw_waveform(test_wave[:, 0], height=300, width=1200, color=(0, 255, 0))
    recon_img = draw_waveform(recon[:, 0], height=300, width=1200, color=(0, 165, 255))
    combined = np.vstack([orig_img, recon_img])
    img_path = os.path.join(folder, f"epoch_{epoch:03d}.png")
    cv2.imwrite(img_path, combined)

    audio_dir = os.path.join(folder, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    recon_audio = normalize_audio(recon[:, 0])
    recon_path = os.path.join(audio_dir, f"epoch_{epoch:03d}_recon.wav")

    if wavfile is not None:
        wavfile.write(recon_path, SAMPLE_RATE, audio_to_int16(recon_audio))

        if epoch == 0:
            orig_audio = normalize_audio(test_wave[:, 0])
            orig_path = os.path.join(audio_dir, f"epoch_{epoch:03d}_orig.wav")
            wavfile.write(orig_path, SAMPLE_RATE, audio_to_int16(orig_audio))

    return img_path, recon_path


def train(model, data, test_wave):
    if len(data) == 0:
        print("[ERROR] No training data. Cannot train.")
        return model

    save_training_outputs(model, test_wave, 0, TRAIN_OUT_DIR)

    print("=== WAVEFORM SNN TRAINING START ===")
    print(f"Epochs: {EPOCHS} | Samples: {len(data)} | Filters: {NUM_FILTERS} | Pool: {POOL_TYPE}")
    print(f"  Waveform: SR={SAMPLE_RATE}, Chunk={CHUNK_SECONDS}s ({CHUNK_SAMPLES} samples)")
    print(f"  SNN: T={T_STEPS}, tau_mem={TAU_MEM}, tau_syn={TAU_SYN}, Dale={USE_DALES}")
    print(f"{'Epoch':<8} | {'Loss':<10} | {'Time':<8} | {'CPU %':<8} | {'RAM %':<8} | {'Mean FR':<8} | {'ETA':<8}")
    print("-" * 80)

    start_time = time.time()
    psutil.cpu_percent(interval=None)

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        indices = np.random.permutation(len(data))
        epoch_loss = 0.0
        epoch_mean_fr = 0.0

        for i in indices:
            loss = model.train_step(data[i])
            if not np.isfinite(loss):
                print(f"  [WARNING] NaN loss at step {i}, skipping remainder.")
                break
            epoch_loss += loss
            mfr, _ = model.get_firing_stats()
            epoch_mean_fr += mfr

        avg_loss = epoch_loss / len(data)
        avg_fr = epoch_mean_fr / len(data)
        if not np.isfinite(avg_loss):
            print("  [ERROR] Epoch produced NaN. Try lowering LEARNING_RATE.")
            break

        epoch_time = time.time() - epoch_start
        elapsed_total = time.time() - start_time
        eta_seconds = epoch_time * (EPOCHS - epoch - 1)
        cpu_percent = psutil.cpu_percent(interval=0)
        ram_percent = psutil.virtual_memory().percent

        print(f"{epoch+1:<8} | {avg_loss:<10.6f} | {epoch_time:<8.2f}s | {cpu_percent:<8.1f} | {ram_percent:<8.1f} | {avg_fr:<8.4f} | {eta_seconds:<8.1f}s")

        img_path, audio_path = save_training_outputs(model, test_wave, epoch + 1, TRAIN_OUT_DIR)
        print(f"  -> img: {img_path}")
        if wavfile is not None:
            print(f"  -> wav: {audio_path}")

    print("-" * 80)
    print(f"Total Time: {elapsed_total/60:.2f} minutes")
    print(f"Final Loss: {avg_loss:.6f} | Final Mean FR: {avg_fr:.4f}")
    return model


def realtime_demo(model):
    print("\n=== REAL-TIME WAVEFORM SNN DEMO ===")
    print("Controls: [t] toggle training | [r] reset weights | [s] save weights | [q] quit")

    q = queue.Queue(maxsize=2)
    training = True
    frame_count = 0
    loss_history = []

    def callback(indata, frames_count, time_info, status):
        audio = indata[:, 0].copy()
        if len(audio) >= CHUNK_SAMPLES:
            q.put(audio[-CHUNK_SAMPLES:])

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=CHUNK_SAMPLES)
    with stream:
        while True:
            try:
                audio = q.get(timeout=0.5)
            except queue.Empty:
                continue

            wave = audio.reshape(CHUNK_SAMPLES, 1)
            recon = model.forward(wave)

            if training:
                loss = 0.5 * np.sum((recon - wave) ** 2)
                if np.isfinite(loss):
                    model.backward(wave)
                    loss_history.append(loss)
                    frame_count += 1
                else:
                    print("  [WARNING] NaN loss, skipping frame.")

            recon_audio = normalize_audio(recon[:, 0], target_peak=0.9)
            sd.play(recon_audio, SAMPLE_RATE, blocking=False)

            orig_img = draw_waveform(wave[:, 0], height=200, width=1200, color=(0, 255, 0))
            recon_img = draw_waveform(recon[:, 0], height=200, width=1200, color=(0, 165, 255))
            display = np.vstack([orig_img, recon_img])

            mfr, mxfr = model.get_firing_stats()
            status = f"TRAIN:{training} | FR:{mfr:.3f} | F:{frame_count}"
            if loss_history:
                status += f" | L:{loss_history[-1]:.4f}"
            cv2.putText(display, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Waveform | SNN Reconstruction (Real-Time)", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                training = not training
                print(f"  Training: {training}")
            elif key == ord('r'):
                model = WaveformSNNAutoencoder(in_channels=IN_CHANNELS)
                loss_history.clear()
                frame_count = 0
                print("  Weights reset.")
            elif key == ord('s'):
                model.save_weights(WEIGHTS_FILE)

    cv2.destroyAllWindows()

    if loss_history:
        avg = np.mean(loss_history[-100:])
        print(f"Final avg loss (last 100 frames): {avg:.6f}")


def demo(model):
    print("Waveform SNN Demo running — press 'q' to quit.")
    q = queue.Queue(maxsize=2)

    def callback(indata, frames_count, time_info, status):
        audio = indata[:, 0].copy()
        if len(audio) >= CHUNK_SAMPLES:
            q.put(audio[-CHUNK_SAMPLES:])

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=CHUNK_SAMPLES)
    with stream:
        while True:
            try:
                audio = q.get(timeout=0.5)
            except queue.Empty:
                continue

            wave = audio.reshape(CHUNK_SAMPLES, 1)
            recon = model.forward(wave)

            recon_audio = normalize_audio(recon[:, 0], target_peak=0.9)
            sd.play(recon_audio, SAMPLE_RATE, blocking=False)

            orig_img = draw_waveform(wave[:, 0], height=200, width=1200, color=(0, 255, 0))
            recon_img = draw_waveform(recon[:, 0], height=200, width=1200, color=(0, 165, 255))
            display = np.vstack([orig_img, recon_img])

            cv2.imshow("Waveform | SNN Reconstructed", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


def save_test_frame(model):
    print("Recording 1s test audio...")
    audio = sd.rec(CHUNK_SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()

    wave = audio.reshape(CHUNK_SAMPLES, 1)
    recon = model.forward(wave)

    orig_img = draw_waveform(wave[:, 0], height=300, width=1200, color=(0, 255, 0))
    recon_img = draw_waveform(recon[:, 0], height=300, width=1200, color=(0, 165, 255))
    combined = np.vstack([orig_img, recon_img])
    cv2.imwrite("waveform_snn_test_frame.png", combined)

    if wavfile is not None:
        recon_audio = normalize_audio(recon[:, 0])
        wavfile.write("waveform_snn_test_frame_recon.wav", SAMPLE_RATE, audio_to_int16(recon_audio))

        orig_audio = normalize_audio(wave[:, 0])
        wavfile.write("waveform_snn_test_frame_orig.wav", SAMPLE_RATE, audio_to_int16(orig_audio))

    print("Saved waveform_snn_test_frame.png + .wav files")


def get_test_frame():
    print("Recording 1s test audio...")
    audio = sd.rec(CHUNK_SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    if len(audio) < CHUNK_SAMPLES:
        print("Audio capture failed — using first dataset sample.")
        return None
    return audio.reshape(CHUNK_SAMPLES, 1)


# ---------- Main Menu ----------
def main():
    print("=" * 60)
    print("  Bio-Inspired SNN Autoencoder — Raw Waveform")
    print("=" * 60)
    print(f"  Resolution : {CHUNK_SAMPLES} samples x {IN_CHANNELS} channel(s)")
    print(f"  Audio      : {SAMPLE_RATE}Hz, {CHUNK_SECONDS}s chunks")
    print(f"  Filters    : {NUM_FILTERS} ({'E/I split' if USE_DALES else 'mixed'})")
    print(f"  Kernel     : {KERNEL_SIZE}x1 (Conv1D)")
    print(f"  Pool       : {POOL_TYPE} (size={WAV_POOL_SIZE})")
    print(f"  SNN Steps  : {T_STEPS} (dt={DT}ms, tau_m={TAU_MEM}ms, tau_s={TAU_SYN}ms)")
    print(f"  LR         : {LEARNING_RATE}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Target FR  : {TARGET_RATE}  |  Decay: {WEIGHT_DECAY}  |  Sparse: {SPARSITY_LAMBDA}")
    print("-" * 60)
    print("  [1] Capture waveform dataset + train + save + test + demo")
    print("  [2] Load waveform dataset      + train + save + test + demo")
    print("  [3] Load weights             + demo (inference only)")
    print("  [4] Load weights             + save test frame")
    print("  [5] REAL-TIME WAVEFORM TRAINING — learn from live mic")
    print("-" * 60)

    choice = input("Choice: ").strip()
    model = WaveformSNNAutoencoder(in_channels=IN_CHANNELS)

    if choice == '1':
        data = capture_waveform_dataset()
        test_wave = get_test_frame()
        if test_wave is None and len(data) > 0:
            test_wave = data[0]
        train(model, data, test_wave)
        model.save_weights(WEIGHTS_FILE)
        save_test_frame(model)
        demo(model)

    elif choice == '2':
        if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
            print("No dataset found. Run option 1 first.")
            return
        data = load_waveform_dataset()
        if len(data) == 0:
            print("[ERROR] Dataset folder exists but contains no valid training samples.")
            print("        Check that .npy files are >= 1 second at 22050 Hz (22050 samples).")
            return
        test_wave = get_test_frame()
        if test_wave is None and len(data) > 0:
            test_wave = data[0]
        train(model, data, test_wave)
        model.save_weights(WEIGHTS_FILE)
        save_test_frame(model)
        demo(model)

    elif choice == '3':
        if not os.path.exists(WEIGHTS_FILE):
            print(f"'{WEIGHTS_FILE}' not found.")
            return
        model.load_weights(WEIGHTS_FILE)
        demo(model)

    elif choice == '4':
        if not os.path.exists(WEIGHTS_FILE):
            print(f"'{WEIGHTS_FILE}' not found.")
            return
        model.load_weights(WEIGHTS_FILE)
        save_test_frame(model)

    elif choice == '5':
        if os.path.exists(WEIGHTS_FILE):
            load = input("Load existing weights? (y/n): ").strip().lower()
            if load == 'y':
                model.load_weights(WEIGHTS_FILE)
        realtime_demo(model)
        model.save_weights(WEIGHTS_FILE)

    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
