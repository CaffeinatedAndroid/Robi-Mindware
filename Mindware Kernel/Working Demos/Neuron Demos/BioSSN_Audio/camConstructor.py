import numpy as np
import cv2
import os
import time
import psutil
import queue
import sounddevice as sd
import librosa

try:
    from scipy.io import wavfile
except ImportError:
    wavfile = None
    print("[WARNING] scipy not found — WAV saving will be disabled.")

# ============ AUDIO CONFIG ============
SAMPLE_RATE     = 22050
CHUNK_SECONDS   = 1.0
CHUNK_SAMPLES   = int(SAMPLE_RATE * CHUNK_SECONDS)
N_FFT           = 512
HOP_LENGTH      = 128
N_MELS          = 128
SPEC_HEIGHT     = 128
SPEC_WIDTH      = 128
IN_CHANNELS     = 1

# ============ SNN CONFIG ============
NUM_FILTERS     = 60
KERNEL_SIZE     = 3
LEARNING_RATE   = 0.0001
EPOCHS          = 10
CAPTURE_SECONDS = 20
DATASET_DIR     = "audio_dataset"
WEIGHTS_FILE    = "audio_snn_weights.bin"
TRAIN_OUT_DIR   = "audio_snn_training_output"
POOL_TYPE       = "avg"

T_STEPS         = 20
DT              = 1.0
TAU_MEM         = 25.0
TAU_SYN         = 5.0
V_TH            = 0.8
V_RESET         = 0.0
MAX_INPUT_RATE  = 0.8
SURROGATE_BETA  = 1.0

WEIGHT_CLIP     = 1.8
WEIGHT_DECAY    = 2e-4
SPARSITY_LAMBDA = 3e-4
TARGET_RATE     = 0.08
HOMEOSTATIC_LR  = 0.007
USE_DALES       = True
GRAD_CLIP       = 1.0
# =======================================


# ---------- Vectorized 2D Convolution ----------
def conv2d(image, kernel, padding=1):
    if padding > 0:
        image = np.pad(image, pad_width=padding, mode='constant', constant_values=0)
    ih, iw = image.shape
    kh, kw = kernel.shape
    out_h = ih - kh + 1
    out_w = iw - kw + 1
    shape = (out_h, out_w, kh, kw)
    strides = (image.strides[0], image.strides[1], image.strides[0], image.strides[1])
    patches = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)
    return np.sum(patches * kernel, axis=(2, 3))


def conv2d_grad_kernel(image, grad_output, k_shape):
    pad = (k_shape[0] - 1) // 2
    padded = np.pad(image, pad_width=pad, mode='constant', constant_values=0)
    return conv2d(padded, grad_output, padding=0)


def conv2d_grad_input(grad_output, kernel, target_shape):
    rot_k = np.rot90(kernel, k=2)
    pad = (kernel.shape[0] - 1) // 2
    padded = np.pad(grad_output, pad_width=pad, mode='constant', constant_values=0)
    result = conv2d(padded, rot_k, padding=0)
    if result.shape != target_shape:
        dh = result.shape[0] - target_shape[0]
        dw = result.shape[1] - target_shape[1]
        result = result[dh//2:dh//2+target_shape[0], dw//2:dw//2+target_shape[1]]
    return result


# ---------- Pooling ----------
def maxpool2x2_forward(x):
    H, W = x.shape
    H -= H % 2; W -= W % 2
    x = x[:H, :W]
    out_h, out_w = H // 2, W // 2
    pooled = np.zeros((out_h, out_w), dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            patch = x[i*2:i*2+2, j*2:j*2+2]
            pooled[i, j] = np.max(patch)
            idx = np.argmax(patch)
            mask[i*2 + idx//2, j*2 + idx%2] = 1.0
    return pooled, mask


def maxpool2x2_backward(d_pooled, mask):
    H, W = mask.shape
    out_h, out_w = H // 2, W // 2
    dx = np.zeros((H, W), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            dx[i*2:i*2+2, j*2:j*2+2] = mask[i*2:i*2+2, j*2:j*2+2] * d_pooled[i, j]
    return dx


def avgpool2x2_forward(x):
    H, W = x.shape
    H -= H % 2; W -= W % 2
    x = x[:H, :W]
    out_h, out_w = H // 2, W // 2
    pooled = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            pooled[i, j] = np.mean(x[i*2:i*2+2, j*2:j*2+2])
    return pooled


def avgpool2x2_backward(d_pooled, target_shape):
    h_p, w_p = d_pooled.shape
    dx = np.zeros(target_shape, dtype=np.float32)
    for i in range(h_p):
        for j in range(w_p):
            dx[i*2:i*2+2, j*2:j*2+2] = d_pooled[i, j] * 0.25
    return dx


def upsample2x2(x, target_shape):
    h, w = x.shape
    out = np.zeros(target_shape, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            out[i*2:i*2+2, j*2:j*2+2] = x[i, j]
    return out


# ---------- Audio Utilities ----------
def normalize_audio(audio, target_peak=0.9):
    peak = np.max(np.abs(audio))
    if peak > 1e-8:
        return np.clip(audio / peak * target_peak, -1.0, 1.0)
    return audio


def audio_to_int16(audio):
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)


# ---------- Audio Preprocessing ----------
def audio_to_spec(audio):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS, center=True
    )
    if mel.shape[1] < SPEC_WIDTH:
        mel = np.pad(mel, ((0, 0), (0, SPEC_WIDTH - mel.shape[1])), mode='edge')
    mel = mel[:, :SPEC_WIDTH]
    if mel.shape[0] < SPEC_HEIGHT:
        mel = np.pad(mel, ((0, SPEC_HEIGHT - mel.shape[0]), (0, 0)), mode='edge')
    mel = mel[:SPEC_HEIGHT, :]

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db + 80.0) / 80.0
    mel_norm = np.clip(mel_norm, 0.0, 1.0)
    return mel_norm.astype(np.float32)[..., np.newaxis]


def spec_to_audio(spec):
    s = spec[..., 0]
    mel_db = s * 80.0 - 80.0
    mel_db = np.clip(mel_db, -80.0, 0.0)
    mel_power = librosa.db_to_power(mel_db)
    audio = librosa.feature.inverse.mel_to_audio(
        mel_power, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    return audio.astype(np.float32)


# ---------- SNN Model ----------
class AudioSNNAutoencoder:
    def __init__(self, in_channels=IN_CHANNELS):
        self.in_channels = in_channels
        fan_in = KERNEL_SIZE * KERNEL_SIZE * in_channels

        self.W_enc = np.random.randn(NUM_FILTERS, KERNEL_SIZE, KERNEL_SIZE, in_channels).astype(np.float32)
        self.W_enc *= np.sqrt(2.0 / fan_in) * 0.5

        if USE_DALES and NUM_FILTERS % 2 == 0:
            self.E_mask = np.ones((NUM_FILTERS, 1, 1, 1), dtype=np.float32)
            self.I_mask = np.ones_like(self.E_mask)
            self.E_mask[NUM_FILTERS//2:] = 0
            self.I_mask[:NUM_FILTERS//2] = 0
            self.W_enc = np.abs(self.W_enc) * self.E_mask - np.abs(self.W_enc) * self.I_mask

        self.W_dec = np.random.randn(NUM_FILTERS, KERNEL_SIZE, KERNEL_SIZE, in_channels).astype(np.float32) * 0.05

        self.b_enc = np.zeros(NUM_FILTERS, dtype=np.float32)
        self.b_dec = np.full(in_channels, 0.5, dtype=np.float32)

        self.alpha = np.exp(-DT / TAU_MEM)
        self.beta  = np.exp(-DT / TAU_SYN)
        self.v_th  = np.full(NUM_FILTERS, V_TH, dtype=np.float32)

        self.x = None
        self.spike_prob = None
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
        x = V_pre - self.v_th[:, None, None]
        return 1.0 / (1.0 + np.abs(SURROGATE_BETA * x))**2

    def forward(self, x):
        self.x = x.astype(np.float32)
        H, W, C = self.x.shape
        assert C == self.in_channels

        self.spike_prob = np.clip(self.x * MAX_INPUT_RATE, 0.0, 1.0)

        V = np.zeros((NUM_FILTERS, H, W), dtype=np.float32)
        I_syn = np.zeros((NUM_FILTERS, H, W), dtype=np.float32)
        S_accum = np.zeros((NUM_FILTERS, H, W), dtype=np.float32)
        V_pre_trace = []

        for t in range(T_STEPS):
            S_in = (np.random.rand(H, W, C) < self.spike_prob).astype(np.float32)
            I_t = np.zeros((NUM_FILTERS, H, W), dtype=np.float32)
            for f in range(NUM_FILTERS):
                for c in range(C):
                    I_t[f] += conv2d(S_in[:, :, c], self.W_enc[f, :, :, c], padding=1)
                I_t[f] += self.b_enc[f]

            I_syn = self.beta * I_syn + I_t
            V_pre = self.alpha * V + I_syn
            spike = (V_pre >= self.v_th[:, None, None]).astype(np.float32)
            V = np.where(spike > 0, V_RESET, V_pre)

            S_accum += spike
            V_pre_trace.append(V_pre.copy())

        self.rate_enc = S_accum / T_STEPS
        self.V_pre_trace = np.array(V_pre_trace)

        self.pooled = np.zeros((NUM_FILTERS, H//2, W//2), dtype=np.float32)
        self.pool_masks = []
        for f in range(NUM_FILTERS):
            if POOL_TYPE == "max":
                p, m = maxpool2x2_forward(self.rate_enc[f])
                self.pooled[f] = p
                self.pool_masks.append(m)
            else:
                self.pooled[f] = avgpool2x2_forward(self.rate_enc[f])
                self.pool_masks.append(None)

        self.upsampled = np.zeros((NUM_FILTERS, H, W), dtype=np.float32)
        for f in range(NUM_FILTERS):
            self.upsampled[f] = upsample2x2(self.pooled[f], (H, W))

        self.output = np.zeros((H, W, C), dtype=np.float32)
        for f in range(NUM_FILTERS):
            for c in range(C):
                self.output[:, :, c] += conv2d(self.upsampled[f], self.W_dec[f, :, :, c], padding=1)
        self.output += self.b_dec
        self.output = np.clip(self.output, -2.0, 2.0)

        return self.output

    def backward(self, target):
        C = self.in_channels
        d_pre = self.output - target

        self.b_dec -= LEARNING_RATE * np.sum(d_pre, axis=(0, 1))
        np.clip(self.b_dec, -2.0, 2.0, out=self.b_dec)

        dW_dec = np.zeros_like(self.W_dec)
        d_upsampled = np.zeros_like(self.upsampled)
        for f in range(NUM_FILTERS):
            for c in range(C):
                dW_dec[f, :, :, c] += conv2d_grad_kernel(
                    self.upsampled[f], d_pre[:, :, c], (KERNEL_SIZE, KERNEL_SIZE)
                )
                d_upsampled[f] += conv2d_grad_input(
                    d_pre[:, :, c], self.W_dec[f, :, :, c], self.upsampled[f].shape
                )

        dW_dec = self._clip_grad(dW_dec)
        self.W_dec -= LEARNING_RATE * dW_dec

        d_pooled = np.zeros_like(self.pooled)
        for f in range(NUM_FILTERS):
            h_p, w_p = self.pooled[f].shape
            for i in range(h_p):
                for j in range(w_p):
                    d_pooled[f, i, j] = np.sum(d_upsampled[f, i*2:i*2+2, j*2:j*2+2])

        d_rate_enc = np.zeros_like(self.rate_enc)
        for f in range(NUM_FILTERS):
            if POOL_TYPE == "max":
                d_rate_enc[f] = maxpool2x2_backward(d_pooled[f], self.pool_masks[f])
            else:
                d_rate_enc[f] = avgpool2x2_backward(d_pooled[f], self.rate_enc[f].shape)

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

        dW_enc = np.zeros_like(self.W_enc)
        for f in range(NUM_FILTERS):
            for c in range(C):
                dW_enc[f, :, :, c] += conv2d_grad_kernel(
                    self.spike_prob[:, :, c], dI_avg[f], (KERNEL_SIZE, KERNEL_SIZE)
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
        n_enc = NUM_FILTERS * KERNEL_SIZE * KERNEL_SIZE * self.in_channels
        n_dec = NUM_FILTERS * KERNEL_SIZE * KERNEL_SIZE * self.in_channels
        self.W_enc = flat[idx:idx+n_enc].reshape(NUM_FILTERS, KERNEL_SIZE, KERNEL_SIZE, self.in_channels)
        idx += n_enc
        self.W_dec = flat[idx:idx+n_dec].reshape(NUM_FILTERS, KERNEL_SIZE, KERNEL_SIZE, self.in_channels)
        idx += n_dec
        self.b_enc = flat[idx:idx+NUM_FILTERS]
        idx += NUM_FILTERS
        self.b_dec = flat[idx:idx+self.in_channels]
        idx += self.in_channels
        self.v_th = flat[idx:idx+NUM_FILTERS]
        print(f"Loaded weights from {path}")


# ---------- Dataset ----------
def capture_audio_dataset():
    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"Recording {CAPTURE_SECONDS}s of audio...")
    total_samples = int(CAPTURE_SECONDS * SAMPLE_RATE)
    recording = sd.rec(total_samples, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    recording = recording.flatten()

    n_chunks = len(recording) // CHUNK_SAMPLES
    specs = []
    for i in range(n_chunks):
        chunk = recording[i*CHUNK_SAMPLES:(i+1)*CHUNK_SAMPLES]
        np.save(os.path.join(DATASET_DIR, f"audio_{i:04d}.npy"), chunk)
        specs.append(audio_to_spec(chunk))

    print(f"Saved {n_chunks} audio chunks to ./{DATASET_DIR}/")
    return specs


def load_audio_dataset():
    files = sorted(f for f in os.listdir(DATASET_DIR) if f.endswith('.npy'))
    specs = []
    for f in files:
        chunk = np.load(os.path.join(DATASET_DIR, f))
        if len(chunk) >= CHUNK_SAMPLES:
            specs.append(audio_to_spec(chunk[:CHUNK_SAMPLES]))
    print(f"Loaded {len(specs)} spectrograms from ./{DATASET_DIR}/")
    return specs


# ---------- Training & Demo ----------
def spec_to_u8(spec):
    s = spec[..., 0] if spec.ndim == 3 else spec
    return (np.clip(s, 0, 1) * 255).astype(np.uint8)


def save_training_outputs(model, test_spec, epoch, folder):
    os.makedirs(folder, exist_ok=True)
    recon = model.forward(test_spec)

    # --- Save spectrogram image ---
    orig_u8 = spec_to_u8(test_spec)
    recon_u8 = spec_to_u8(recon)
    orig_bgr = cv2.cvtColor(orig_u8, cv2.COLOR_GRAY2BGR)
    recon_bgr = cv2.cvtColor(recon_u8, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([orig_bgr, recon_bgr])
    img_path = os.path.join(folder, f"epoch_{epoch:03d}.png")
    cv2.imwrite(img_path, combined)

    # --- Save audio files ---
    audio_dir = os.path.join(folder, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    recon_audio = spec_to_audio(recon)
    recon_audio = normalize_audio(recon_audio)
    recon_path = os.path.join(audio_dir, f"epoch_{epoch:03d}_recon.wav")

    if wavfile is not None:
        wavfile.write(recon_path, SAMPLE_RATE, audio_to_int16(recon_audio))

        if epoch == 0:
            orig_audio = spec_to_audio(test_spec)
            orig_audio = normalize_audio(orig_audio)
            orig_path = os.path.join(audio_dir, f"epoch_{epoch:03d}_orig.wav")
            wavfile.write(orig_path, SAMPLE_RATE, audio_to_int16(orig_audio))

    return img_path, recon_path


def train(model, data, test_spec):
    save_training_outputs(model, test_spec, 0, TRAIN_OUT_DIR)

    print("=== AUDIO SNN TRAINING START ===")
    print(f"Epochs: {EPOCHS} | Samples: {len(data)} | Filters: {NUM_FILTERS} | Pool: {POOL_TYPE}")
    print(f"  Audio: SR={SAMPLE_RATE}, Chunk={CHUNK_SECONDS}s, Mel={N_MELS}, Hop={HOP_LENGTH}")
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

        img_path, audio_path = save_training_outputs(model, test_spec, epoch + 1, TRAIN_OUT_DIR)
        print(f"  -> img: {img_path}")
        if wavfile is not None:
            print(f"  -> wav: {audio_path}")

    print("-" * 80)
    print(f"Total Time: {elapsed_total/60:.2f} minutes")
    print(f"Final Loss: {avg_loss:.6f} | Final Mean FR: {avg_fr:.4f}")
    return model


def realtime_demo(model):
    print("\n=== REAL-TIME AUDIO SNN DEMO ===")
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

            spec = audio_to_spec(audio)
            recon = model.forward(spec)

            if training:
                loss = 0.5 * np.sum((recon - spec) ** 2)
                if np.isfinite(loss):
                    model.backward(spec)
                    loss_history.append(loss)
                    frame_count += 1
                else:
                    print("  [WARNING] NaN loss, skipping frame.")

            # --- Play reconstructed audio live ---
            recon_audio = spec_to_audio(recon)
            recon_audio = normalize_audio(recon_audio, target_peak=0.9)
            sd.play(recon_audio, SAMPLE_RATE, blocking=False)

            # --- Visualize ---
            orig_u8 = spec_to_u8(spec)
            recon_u8 = spec_to_u8(recon)
            orig_big = cv2.resize(orig_u8, (300, 300), interpolation=cv2.INTER_NEAREST)
            recon_big = cv2.resize(recon_u8, (300, 300), interpolation=cv2.INTER_NEAREST)
            orig_bgr = cv2.cvtColor(orig_big, cv2.COLOR_GRAY2BGR)
            recon_bgr = cv2.cvtColor(recon_big, cv2.COLOR_GRAY2BGR)

            mfr, mxfr = model.get_firing_stats()
            status = f"TRAIN:{training} | FR:{mfr:.3f} | F:{frame_count}"
            if loss_history:
                status += f" | L:{loss_history[-1]:.4f}"
            cv2.putText(orig_bgr, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Audio Spectrogram | SNN Reconstruction (Real-Time)", np.hstack([orig_bgr, recon_bgr]))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                training = not training
                print(f"  Training: {training}")
            elif key == ord('r'):
                model = AudioSNNAutoencoder(in_channels=IN_CHANNELS)
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
    print("Audio SNN Demo running — press 'q' to quit.")
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

            spec = audio_to_spec(audio)
            recon = model.forward(spec)

            # --- Play reconstructed audio live ---
            recon_audio = spec_to_audio(recon)
            recon_audio = normalize_audio(recon_audio, target_peak=0.9)
            sd.play(recon_audio, SAMPLE_RATE, blocking=False)

            # --- Visualize ---
            orig_u8 = spec_to_u8(spec)
            recon_u8 = spec_to_u8(recon)
            orig_big = cv2.resize(orig_u8, (300, 300), interpolation=cv2.INTER_NEAREST)
            recon_big = cv2.resize(recon_u8, (300, 300), interpolation=cv2.INTER_NEAREST)
            orig_bgr = cv2.cvtColor(orig_big, cv2.COLOR_GRAY2BGR)
            recon_bgr = cv2.cvtColor(recon_big, cv2.COLOR_GRAY2BGR)

            cv2.imshow("Audio Spectrogram | SNN Reconstructed", np.hstack([orig_bgr, recon_bgr]))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


def save_test_frame(model):
    print("Recording 1s test audio...")
    audio = sd.rec(CHUNK_SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()

    spec = audio_to_spec(audio)
    recon = model.forward(spec)

    # Save image
    orig_u8 = spec_to_u8(spec)
    recon_u8 = spec_to_u8(recon)
    orig_bgr = cv2.cvtColor(orig_u8, cv2.COLOR_GRAY2BGR)
    recon_bgr = cv2.cvtColor(recon_u8, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([orig_bgr, recon_bgr])
    cv2.imwrite("audio_snn_test_frame.png", combined)

    # Save audio
    if wavfile is not None:
        recon_audio = spec_to_audio(recon)
        recon_audio = normalize_audio(recon_audio)
        wavfile.write("audio_snn_test_frame_recon.wav", SAMPLE_RATE, audio_to_int16(recon_audio))

        orig_audio = spec_to_audio(spec)
        orig_audio = normalize_audio(orig_audio)
        wavfile.write("audio_snn_test_frame_orig.wav", SAMPLE_RATE, audio_to_int16(orig_audio))

    print("Saved audio_snn_test_frame.png + .wav files")


def get_test_frame():
    print("Recording 1s test audio...")
    audio = sd.rec(CHUNK_SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    if len(audio) < CHUNK_SAMPLES:
        print("Audio capture failed — using first dataset sample.")
        return None
    return audio_to_spec(audio)


# ---------- Main Menu ----------
def main():
    print("=" * 60)
    print("  Bio-Inspired SNN Autoencoder — Audio (Mel-Spectrogram)")
    print("=" * 60)
    print(f"  Resolution : {SPEC_HEIGHT}x{SPEC_WIDTH}x{IN_CHANNELS} (mel-spec)")
    print(f"  Audio      : {SAMPLE_RATE}Hz, {CHUNK_SECONDS}s chunks")
    print(f"  Filters    : {NUM_FILTERS} ({'E/I split' if USE_DALES else 'mixed'})")
    print(f"  Kernel     : {KERNEL_SIZE}x{KERNEL_SIZE}")
    print(f"  Pool       : {POOL_TYPE}")
    print(f"  SNN Steps  : {T_STEPS} (dt={DT}ms, tau_m={TAU_MEM}ms, tau_s={TAU_SYN}ms)")
    print(f"  LR         : {LEARNING_RATE}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Target FR  : {TARGET_RATE}  |  Decay: {WEIGHT_DECAY}  |  Sparse: {SPARSITY_LAMBDA}")
    print("-" * 60)
    print("  [1] Capture audio dataset + train + save + test + demo")
    print("  [2] Load audio dataset      + train + save + test + demo")
    print("  [3] Load weights          + demo (inference only)")
    print("  [4] Load weights          + save test frame")
    print("  [5] REAL-TIME AUDIO TRAINING — learn from live mic")
    print("-" * 60)

    choice = input("Choice: ").strip()
    model = AudioSNNAutoencoder(in_channels=IN_CHANNELS)

    if choice == '1':
        data = capture_audio_dataset()
        test_spec = get_test_frame()
        if test_spec is None and len(data) > 0:
            test_spec = data[0]
        train(model, data, test_spec)
        model.save_weights(WEIGHTS_FILE)
        save_test_frame(model)
        demo(model)

    elif choice == '2':
        if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
            print("No dataset found. Run option 1 first.")
            return
        data = load_audio_dataset()
        test_spec = get_test_frame()
        if test_spec is None and len(data) > 0:
            test_spec = data[0]
        train(model, data, test_spec)
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
