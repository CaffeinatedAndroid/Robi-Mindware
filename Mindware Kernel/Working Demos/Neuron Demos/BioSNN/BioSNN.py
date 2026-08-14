import numpy as np
import cv2
import os
import time
import psutil

# ============ GLOBAL CONFIG ============
IMG_SIZE        = 64
NUM_FILTERS     = 32
KERNEL_SIZE     = 3
LEARNING_RATE   = 0.00015      # Lowered for spiking dynamics
EPOCHS          = 25
CAPTURE_SECONDS = 10
DATASET_DIR     = "dataset"
WEIGHTS_FILE    = "snn_weights.bin"
TRAIN_OUT_DIR   = "snn_training_output"
POOL_TYPE       = "avg"

# --- SNN / Biological neuron settings ---
T_STEPS         = 12          # temporal steps per image (simulation depth)
DT              = 1.0         # ms per time step
TAU_MEM         = 20.0        # membrane time constant (ms)
TAU_SYN         = 5.0         # synaptic time constant (ms)
V_TH            = 1.0         # base firing threshold
V_RESET         = 0.0         # reset potential after spike
MAX_INPUT_RATE  = 0.8         # max Poisson firing probability per step (0-1)
SURROGATE_BETA  = 1.0         # steepness of surrogate gradient

WEIGHT_CLIP     = 1.5
WEIGHT_DECAY    = 1e-4
SPARSITY_LAMBDA = 8e-4
TARGET_RATE     = 0.08        # target firing rate (fraction of T_STEPS)
HOMEOSTATIC_LR  = 0.005
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


# ---------- SNN Model ----------
class SNNAutoencoder:
    """
    Spiking Autoencoder with LIF encoder and linear rate decoder.

    Encoder:
        Poisson(input pixels) -> Conv2D synaptic current -> LIF neurons
        -> accumulate spikes over T_STEPS -> rate map -> pool -> bottleneck

    Decoder:
        Upsample bottleneck rates -> Conv2D -> reconstruction
        (Linear decoder for stable gradient flow to the spiking bottleneck.)

    Learning:
        Single-step surrogate gradient through the LIF threshold.
        Homeostatic bias + adaptive threshold plasticity.
        Dale's principle + weight decay + sparsity.
    """
    def __init__(self):
        fan_in = KERNEL_SIZE * KERNEL_SIZE * 3
        self.W_enc = np.random.randn(NUM_FILTERS, KERNEL_SIZE, KERNEL_SIZE, 3).astype(np.float32)
        self.W_enc *= np.sqrt(2.0 / fan_in) * 0.5

        if USE_DALES and NUM_FILTERS % 2 == 0:
            self.E_mask = np.ones((NUM_FILTERS, 1, 1, 1), dtype=np.float32)
            self.I_mask = np.ones_like(self.E_mask)
            self.E_mask[NUM_FILTERS//2:] = 0
            self.I_mask[:NUM_FILTERS//2] = 0
            self.W_enc = np.abs(self.W_enc) * self.E_mask - np.abs(self.W_enc) * self.I_mask

        self.W_dec = np.random.randn(NUM_FILTERS, KERNEL_SIZE, KERNEL_SIZE, 3).astype(np.float32) * 0.05

        self.b_enc = np.zeros(NUM_FILTERS, dtype=np.float32)
        self.b_dec = np.full(3, 0.5, dtype=np.float32)

        # LIF time constants
        self.alpha = np.exp(-DT / TAU_MEM)
        self.beta  = np.exp(-DT / TAU_SYN)
        # Per-filter adaptive threshold (homeostasis)
        self.v_th  = np.full(NUM_FILTERS, V_TH, dtype=np.float32)

        # Cached states for backprop
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
        """
        Fast sigmoid surrogate for the spike gradient.
        d(spike)/dV ≈ 1 / (1 + |beta*(V - v_th)|)^2
        """
        x = V_pre - self.v_th[:, None, None]
        return 1.0 / (1.0 + np.abs(SURROGATE_BETA * x))**2

    def forward(self, x):
        self.x = x.astype(np.float32)
        H, W, C = self.x.shape

        # ---- 1. Poisson input encoding ----
        # Convert pixel intensity (0-1) to spike probability per time step
        self.spike_prob = np.clip(self.x * MAX_INPUT_RATE, 0.0, 1.0)

        # ---- 2. Encoder SNN: T_STEPS of LIF dynamics ----
        V = np.zeros((NUM_FILTERS, H, W), dtype=np.float32)
        I_syn = np.zeros((NUM_FILTERS, H, W), dtype=np.float32)
        S_accum = np.zeros((NUM_FILTERS, H, W), dtype=np.float32)
        V_pre_trace = []

        for t in range(T_STEPS):
            # Poisson spike sample: 1 where rand < prob, else 0
            S_in = (np.random.rand(H, W, C) < self.spike_prob).astype(np.float32)

            # Synaptic current from input spikes (convolution across channels)
            I_t = np.zeros((NUM_FILTERS, H, W), dtype=np.float32)
            for f in range(NUM_FILTERS):
                for c in range(C):
                    I_t[f] += conv2d(S_in[:, :, c], self.W_enc[f, :, :, c], padding=1)
                I_t[f] += self.b_enc[f]

            # LIF dynamics
            I_syn = self.beta * I_syn + I_t
            V_pre = self.alpha * V + I_syn
            spike = (V_pre >= self.v_th[:, None, None]).astype(np.float32)
            V = np.where(spike > 0, V_RESET, V_pre)

            S_accum += spike
            V_pre_trace.append(V_pre.copy())

        # Time-averaged firing rates become the "activation map"
        self.rate_enc = S_accum / T_STEPS
        self.V_pre_trace = np.array(V_pre_trace)  # (T, F, H, W)

        # ---- 3. Pooling (operates on spike rates) ----
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

        # ---- 4. Upsample bottleneck rates ----
        self.upsampled = np.zeros((NUM_FILTERS, H, W), dtype=np.float32)
        for f in range(NUM_FILTERS):
            self.upsampled[f] = upsample2x2(self.pooled[f], (H, W))

        # ---- 5. Decoder: linear rate reconstruction ----
        self.output = np.zeros((H, W, C), dtype=np.float32)
        for f in range(NUM_FILTERS):
            for c in range(C):
                self.output[:, :, c] += conv2d(self.upsampled[f], self.W_dec[f, :, :, c], padding=1)
        self.output += self.b_dec
        self.output = np.clip(self.output, -2.0, 2.0)

        return self.output

    def backward(self, target):
        C = 3
        d_pre = self.output - target

        # ---- Decoder gradients (standard conv backprop on rates) ----
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

        # ---- Upsample backprop ----
        d_pooled = np.zeros_like(self.pooled)
        for f in range(NUM_FILTERS):
            h_p, w_p = self.pooled[f].shape
            for i in range(h_p):
                for j in range(w_p):
                    d_pooled[f, i, j] = np.sum(d_upsampled[f, i*2:i*2+2, j*2:j*2+2])

        # ---- Pool backprop to rate_enc ----
        d_rate_enc = np.zeros_like(self.rate_enc)
        for f in range(NUM_FILTERS):
            if POOL_TYPE == "max":
                d_rate_enc[f] = maxpool2x2_backward(d_pooled[f], self.pool_masks[f])
            else:
                d_rate_enc[f] = avgpool2x2_backward(d_pooled[f], self.rate_enc[f].shape)

        # ---- Sparsity penalty on encoder firing rates ----
        d_rate_enc += SPARSITY_LAMBDA * np.sign(self.rate_enc)

        # ---- Homeostatic plasticity: bias adapts to push mean rate toward target ----
        for f in range(NUM_FILTERS):
            mean_rate = np.mean(self.rate_enc[f])
            error = mean_rate - TARGET_RATE
            self.b_enc[f] -= HOMEOSTATIC_LR * error
        np.clip(self.b_enc, -2.0, 2.0, out=self.b_enc)

        # ---- Adaptive threshold: intrinsic plasticity ----
        for f in range(NUM_FILTERS):
            mean_rate = np.mean(self.rate_enc[f])
            self.v_th[f] += 0.01 * (mean_rate - TARGET_RATE)
        self.v_th = np.clip(self.v_th, 0.5, 2.0)

        # ---- Surrogate gradient through LIF spike nonlinearity ----
        # We use the fast-sigmoid surrogate averaged over the time window.
        # The temporal chain through alpha/beta is omitted for efficiency
        # (single-step surrogate gradient, standard in SNN training).
        sg = np.zeros_like(self.rate_enc)
        for t in range(T_STEPS):
            sg += self.surrogate_grad(self.V_pre_trace[t])
        sg /= T_STEPS

        dI_avg = d_rate_enc * sg  # gradient w.r.t. average synaptic current

        # ---- Encoder weight gradients ----
        # Using the expected Poisson rate (spike_prob) for low-variance gradients.
        dW_enc = np.zeros_like(self.W_enc)
        for f in range(NUM_FILTERS):
            for c in range(C):
                dW_enc[f, :, :, c] += conv2d_grad_kernel(
                    self.spike_prob[:, :, c], dI_avg[f], (KERNEL_SIZE, KERNEL_SIZE)
                )

        dW_enc = self._clip_grad(dW_enc)
        self.W_enc -= LEARNING_RATE * dW_enc

        # ---- Biological constraints ----
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
        """Return mean and max encoder firing rates for monitoring."""
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
        n_enc = NUM_FILTERS * KERNEL_SIZE * KERNEL_SIZE * 3
        n_dec = NUM_FILTERS * KERNEL_SIZE * KERNEL_SIZE * 3
        self.W_enc = flat[idx:idx+n_enc].reshape(NUM_FILTERS, KERNEL_SIZE, KERNEL_SIZE, 3)
        idx += n_enc
        self.W_dec = flat[idx:idx+n_dec].reshape(NUM_FILTERS, KERNEL_SIZE, KERNEL_SIZE, 3)
        idx += n_dec
        self.b_enc = flat[idx:idx+NUM_FILTERS]
        idx += NUM_FILTERS
        self.b_dec = flat[idx:idx+3]
        idx += 3
        self.v_th = flat[idx:idx+NUM_FILTERS]
        print(f"Loaded weights from {path}")


# ---------- Dataset ----------
def capture_dataset():
    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"Capturing for {CAPTURE_SECONDS}s — press 'q' to stop early.")
    cap = cv2.VideoCapture(0)
    frames = []
    t0 = time.time()
    while time.time() - t0 < CAPTURE_SECONDS:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        norm = small.astype(np.float32) / 255.0
        frames.append(norm)
        cv2.imshow("Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    for i, f in enumerate(frames):
        np.save(os.path.join(DATASET_DIR, f"frame_{i:04d}.npy"), f)
    print(f"Saved {len(frames)} frames to ./{DATASET_DIR}/")
    return frames


def load_dataset():
    files = sorted(f for f in os.listdir(DATASET_DIR) if f.endswith('.npy'))
    frames = [np.load(os.path.join(DATASET_DIR, f)) for f in files]
    print(f"Loaded {len(frames)} frames from ./{DATASET_DIR}/")
    return frames


# ---------- Training & Demo ----------
def save_training_image(model, test_frame, epoch, folder):
    os.makedirs(folder, exist_ok=True)
    recon = model.forward(test_frame)
    recon_u8 = (np.clip(recon, 0, 1) * 255).astype(np.uint8)
    orig_u8  = (np.clip(test_frame, 0, 1) * 255).astype(np.uint8)
    combined = np.hstack([orig_u8, recon_u8])
    path = os.path.join(folder, f"epoch_{epoch:03d}.png")
    cv2.imwrite(path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    return path


def train(model, data, test_frame):
    save_training_image(model, test_frame, 0, TRAIN_OUT_DIR)

    print("=== SNN TRAINING START ===")
    print(f"Epochs: {EPOCHS} | Images: {len(data)} | Filters: {NUM_FILTERS} | Pool: {POOL_TYPE} | LR: {LEARNING_RATE}")
    print(f"  SNN: T={T_STEPS}, tau_mem={TAU_MEM}, tau_syn={TAU_SYN}, v_th={V_TH}, input_rate={MAX_INPUT_RATE}")
    print(f"  Bio: decay={WEIGHT_DECAY}, sparse={SPARSITY_LAMBDA}, target_rate={TARGET_RATE}, Dale={USE_DALES}")
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
                print(f"  [WARNING] NaN loss at step {i}, skipping remainder of epoch.")
                break
            epoch_loss += loss
            mfr, _ = model.get_firing_stats()
            epoch_mean_fr += mfr

        avg_loss = epoch_loss / len(data)
        avg_fr = epoch_mean_fr / len(data)
        if not np.isfinite(avg_loss):
            print("  [ERROR] Epoch produced NaN. Try lowering LEARNING_RATE or T_STEPS.")
            break

        epoch_time = time.time() - epoch_start
        elapsed_total = time.time() - start_time
        eta_seconds = epoch_time * (EPOCHS - epoch - 1)
        cpu_percent = psutil.cpu_percent(interval=0)
        ram_percent = psutil.virtual_memory().percent

        print(f"{epoch+1:<8} | {avg_loss:<10.6f} | {epoch_time:<8.2f}s | {cpu_percent:<8.1f} | {ram_percent:<8.1f} | {avg_fr:<8.4f} | {eta_seconds:<8.1f}s")

        img_path = save_training_image(model, test_frame, epoch + 1, TRAIN_OUT_DIR)
        print(f"  -> saved progress: {img_path}")

    print("-" * 80)
    print(f"Total Time: {elapsed_total/60:.2f} minutes")
    print(f"Final Loss: {avg_loss:.6f} | Final Mean FR: {avg_fr:.4f}")
    return model


def realtime_demo(model):
    print("\n=== REAL-TIME SNN TRAINING DEMO ===")
    print("Controls: [t] toggle training | [r] reset weights | [s] save weights | [q] quit")

    cap = cv2.VideoCapture(0)
    training = True
    frame_count = 0
    loss_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        norm = small.astype(np.float32) / 255.0

        recon = model.forward(norm)

        if training:
            loss = 0.5 * np.sum((recon - norm) ** 2)
            if np.isfinite(loss):
                model.backward(norm)
                loss_history.append(loss)
                frame_count += 1
            else:
                print("  [WARNING] NaN loss in real-time mode, skipping frame.")

        recon_u8 = (np.clip(recon, 0, 1) * 255).astype(np.uint8)
        recon_bgr = cv2.cvtColor(recon_u8, cv2.COLOR_RGB2BGR)

        left = cv2.resize(frame, (300, 300), interpolation=cv2.INTER_NEAREST)
        right = cv2.resize(recon_bgr, (300, 300), interpolation=cv2.INTER_NEAREST)

        mfr, mxfr = model.get_firing_stats()
        status = f"TRAIN:{training} | FR:{mfr:.3f} | F:{frame_count}"
        if loss_history:
            status += f" | L:{loss_history[-1]:.4f}"
        cv2.putText(left, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Original | SNN Reconstruction (Real-Time)", np.hstack([left, right]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            training = not training
            print(f"  Training: {training}")
        elif key == ord('r'):
            model = SNNAutoencoder()
            loss_history.clear()
            frame_count = 0
            print("  Weights reset.")
        elif key == ord('s'):
            model.save_weights(WEIGHTS_FILE)

    cap.release()
    cv2.destroyAllWindows()

    if loss_history:
        avg = np.mean(loss_history[-100:])
        print(f"Final avg loss (last 100 frames): {avg:.6f}")


def demo(model):
    print("SNN Demo running — press 'q' to quit.")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        norm = small.astype(np.float32) / 255.0
        recon = model.forward(norm)
        recon_u8 = (np.clip(recon, 0, 1) * 255).astype(np.uint8)
        recon_bgr = cv2.cvtColor(recon_u8, cv2.COLOR_RGB2BGR)
        left = cv2.resize(frame, (300, 300), interpolation=cv2.INTER_NEAREST)
        right = cv2.resize(recon_bgr, (300, 300), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Original | SNN Reconstructed", np.hstack([left, right]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def save_test_frame(model):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Camera failed.")
        return
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    norm = small.astype(np.float32) / 255.0
    recon = model.forward(norm)
    recon_u8 = (np.clip(recon, 0, 1) * 255).astype(np.uint8)
    combined = np.hstack([small, recon_u8])
    cv2.imwrite("snn_test_frame.png", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print("Saved snn_test_frame.png")


def get_test_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Camera failed — using first dataset frame.")
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    return small.astype(np.float32) / 255.0


# ---------- Main Menu ----------
def main():
    print("=" * 60)
    print("  Bio-Inspired SNN Autoencoder — Pure NumPy")
    print("=" * 60)
    print(f"  Resolution : {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Filters    : {NUM_FILTERS} ({'E/I split' if USE_DALES else 'mixed'})")
    print(f"  Kernel     : {KERNEL_SIZE}x{KERNEL_SIZE}")
    print(f"  Pool       : {POOL_TYPE}")
    print(f"  SNN Steps  : {T_STEPS} (dt={DT}ms, tau_m={TAU_MEM}ms, tau_s={TAU_SYN}ms)")
    print(f"  LR         : {LEARNING_RATE}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Target FR  : {TARGET_RATE}  |  Decay: {WEIGHT_DECAY}  |  Sparse: {SPARSITY_LAMBDA}")
    print("-" * 60)
    print("  [1] Capture dataset + train + save + test + demo")
    print("  [2] Load dataset      + train + save + test + demo")
    print("  [3] Load weights      + demo (inference only)")
    print("  [4] Load weights      + save test frame")
    print("  [5] REAL-TIME SNN TRAINING — learn from live camera")
    print("-" * 60)

    choice = input("Choice: ").strip()
    model = SNNAutoencoder()

    if choice == '1':
        data = capture_dataset()
        test_frame = get_test_frame()
        if test_frame is None and len(data) > 0:
            test_frame = data[0]
        train(model, data, test_frame)
        model.save_weights(WEIGHTS_FILE)
        save_test_frame(model)
        demo(model)

    elif choice == '2':
        if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
            print("No dataset found. Run option 1 first.")
            return
        data = load_dataset()
        test_frame = get_test_frame()
        if test_frame is None and len(data) > 0:
            test_frame = data[0]
        train(model, data, test_frame)
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
