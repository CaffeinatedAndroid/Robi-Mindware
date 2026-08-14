import numpy as np
import cv2
import os
import time
import psutil

# ============ VIEWER CONFIG ============
GRID_COLS        = 4        # Feature maps per row
FPS_SMOOTH       = 0.9      # Exponential moving average for FPS
SHOW_WEIGHTS     = True     # Toggle decoder weight overlay
COLORMAP         = cv2.COLORMAP_VIRIDIS
# =======================================


# ============ VIEWER LAYOUT CONSTANTS (50% SCALE) ============
STATS_W        = 110
MAIN_W         = 960 - STATS_W          # 850
MAIN_H         = 540
BG_COLOR       = (20, 20, 25)

# Small reference previews (top-left corner)
PREVIEW_W      = 170
PREVIEW_H      = 128
PREVIEW_BAR_H  = 13
PREVIEW_TOTAL_H = PREVIEW_H + PREVIEW_BAR_H   # 141

# Large neuron panels (everything below the previews)
NEURON_W       = MAIN_W // 2             # 425
NEURON_H       = MAIN_H - PREVIEW_TOTAL_H # 399
NEURON_BAR_H   = 13
NEURON_CONTENT_H = NEURON_H - NEURON_BAR_H  # 386
# =============================================================



# ============ GLOBAL CONFIG ============
IMG_SIZE        = 32
NUM_FILTERS     = 16
KERNEL_SIZE     = 3
LEARNING_RATE   = 0.001
EPOCHS          = 5
CAPTURE_SECONDS = 10
DATASET_DIR     = "dataset"
WEIGHTS_FILE    = "weights.bin"
TRAIN_OUT_DIR   = "training_output"
POOL_TYPE       = "avg"

# --- Biological neuron settings ---
LEAK            = 0.02
WEIGHT_CLIP     = 1.5
WEIGHT_DECAY    = 5e-5
SPARSITY_LAMBDA = 5e-4
TARGET_RATE     = 0.10
HOMEOSTATIC_LR  = 0.005
USE_DALES       = True
GRAD_CLIP       = 1.0
# =======================================


# ---------- Activations ----------
def leaky_relu(x, leak=LEAK):
    return np.where(x > 0, x, leak * x)

def leaky_relu_deriv(x, leak=LEAK):
    return np.where(x > 0, 1.0, leak)


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


# ---------- Model ----------
class BioAutoencoder:
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

        self.x = None
        self.conv_enc = None
        self.act_enc = None
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
                              (self.b_enc, "b_enc"), (self.b_dec, "b_dec")]:
            if not np.isfinite(tensor).all():
                print(f"  [WARNING] NaN/Inf detected in {label} after {name}!")
                return True
        return False

    def forward(self, x):
        self.x = x.astype(np.float32)
        H, W, C = self.x.shape

        self.conv_enc = np.zeros((NUM_FILTERS, H, W), dtype=np.float32)
        for f in range(NUM_FILTERS):
            for c in range(C):
                self.conv_enc[f] += conv2d(self.x[:, :, c], self.W_enc[f, :, :, c], padding=1)
            self.conv_enc[f] += self.b_enc[f]

        self.act_enc = leaky_relu(self.conv_enc)

        self.pooled = np.zeros((NUM_FILTERS, H//2, W//2), dtype=np.float32)
        self.pool_masks = []
        for f in range(NUM_FILTERS):
            if POOL_TYPE == "max":
                p, m = maxpool2x2_forward(self.act_enc[f])
                self.pooled[f] = p
                self.pool_masks.append(m)
            else:
                self.pooled[f] = avgpool2x2_forward(self.act_enc[f])
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
        C = 3
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

        d_act = np.zeros_like(self.act_enc)
        for f in range(NUM_FILTERS):
            if POOL_TYPE == "max":
                d_act[f] = maxpool2x2_backward(d_pooled[f], self.pool_masks[f])
            else:
                d_act[f] = avgpool2x2_backward(d_pooled[f], self.act_enc[f].shape)

        d_conv_enc = d_act * leaky_relu_deriv(self.conv_enc)

        for f in range(NUM_FILTERS):
            mean_rate = np.mean(self.act_enc[f])
            error = mean_rate - TARGET_RATE
            self.b_enc[f] -= HOMEOSTATIC_LR * error
        np.clip(self.b_enc, -2.0, 2.0, out=self.b_enc)

        d_conv_enc += SPARSITY_LAMBDA * np.sign(self.act_enc)

        dW_enc = np.zeros_like(self.W_enc)
        for f in range(NUM_FILTERS):
            for c in range(C):
                dW_enc[f, :, :, c] += conv2d_grad_kernel(
                    self.x[:, :, c], d_conv_enc[f], (KERNEL_SIZE, KERNEL_SIZE)
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

    def save_weights(self, path):
        flat = np.concatenate([
            self.W_enc.flatten(),
            self.W_dec.flatten(),
            self.b_enc.flatten(),
            self.b_dec.flatten()
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
    print("=== TRAINING START ===")
    print(f"Epochs: {EPOCHS} | Images: {len(data)} | Filters: {NUM_FILTERS} | Pool: {POOL_TYPE} | LR: {LEARNING_RATE}")
    print(f"  Bio: leak={LEAK}, decay={WEIGHT_DECAY}, sparse={SPARSITY_LAMBDA}, target_rate={TARGET_RATE}, Dale={USE_DALES}")
    print(f"{'Epoch':<8} | {'Loss':<10} | {'Time':<8} | {'CPU %':<8} | {'RAM %':<8} | {'ETA':<8}")
    print("-" * 65)

    start_time = time.time()
    psutil.cpu_percent(interval=None)

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        indices = np.random.permutation(len(data))
        epoch_loss = 0.0

        for i in indices:
            loss = model.train_step(data[i])
            if not np.isfinite(loss):
                print(f"  [WARNING] NaN loss at step {i}, skipping remainder of epoch.")
                break
            epoch_loss += loss

        avg_loss = epoch_loss / len(data)
        if not np.isfinite(avg_loss):
            print("  [ERROR] Epoch produced NaN. Try lowering LEARNING_RATE.")
            break

        epoch_time = time.time() - epoch_start
        elapsed_total = time.time() - start_time
        eta_seconds = epoch_time * (EPOCHS - epoch - 1)
        cpu_percent = psutil.cpu_percent(interval=0)
        ram_percent = psutil.virtual_memory().percent

        print(f"{epoch+1:<8} | {avg_loss:<10.6f} | {epoch_time:<8.2f}s | {cpu_percent:<8.1f} | {ram_percent:<8.1f} | {eta_seconds:<8.1f}s")
        img_path = save_training_image(model, test_frame, epoch + 1, TRAIN_OUT_DIR)
        print(f"  -> saved progress: {img_path}")

    print("-" * 65)
    print(f"Total Time: {elapsed_total/60:.2f} minutes")
    print(f"Final Loss: {avg_loss:.6f}")
    return model


def realtime_demo(model):
    print("\n=== REAL-TIME TRAINING DEMO ===")
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

        status = f"TRAIN: {training} | Frames: {frame_count}"
        if loss_history:
            status += f" | Loss: {loss_history[-1]:.4f}"
        cv2.putText(left, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Original | Reconstructed (Real-Time)", np.hstack([left, right]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            training = not training
            print(f"  Training: {training}")
        elif key == ord('r'):
            model = BioAutoencoder()
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
    print("Demo running — press 'q' to quit.")
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
        cv2.imshow("Original | Reconstructed", np.hstack([left, right]))
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
    cv2.imwrite("test_frame.png", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print("Saved test_frame.png")


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


# ============================================================
# LIVE NEURON VIEWER
# ============================================================

def normalize_for_display(tensor, target_range=(0, 255)):
    """Normalize any tensor to 0-255 uint8 safely."""
    t = np.array(tensor, dtype=np.float32)
    t_min, t_max = np.min(t), np.max(t)
    if t_max - t_min < 1e-8:
        t = np.zeros_like(t)
    else:
        t = (t - t_min) / (t_max - t_min)
    t = (t * (target_range[1] - target_range[0]) + target_range[0])
    return np.clip(t, 0, 255).astype(np.uint8)


def tile_images(images, cols=None, border=1, border_color=40):
    """Tile a list of 2D or RGB arrays into a grid with borders."""
    if not images:
        return np.zeros((64, 64), dtype=np.uint8)

    n = len(images)
    if cols is None:
        cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    # Normalize if not already uint8
    norm_images = []
    for im in images:
        if im.dtype != np.uint8:
            im = normalize_for_display(im, (0, 255))
        norm_images.append(im)

    max_h = max(im.shape[0] for im in norm_images)
    max_w = max(im.shape[1] for im in norm_images)
    is_rgb = (len(norm_images[0].shape) == 3 and norm_images[0].shape[2] == 3)

    canvas_h = rows * max_h + (rows + 1) * border
    canvas_w = cols * max_w + (cols + 1) * border

    if is_rgb:
        canvas = np.full((canvas_h, canvas_w, 3), border_color, dtype=np.uint8)
    else:
        canvas = np.full((canvas_h, canvas_w), border_color, dtype=np.uint8)

    for idx, im in enumerate(norm_images):
        r, c = divmod(idx, cols)
        y = r * max_h + (r + 1) * border
        x = c * max_w + (c + 1) * border
        dy = (max_h - im.shape[0]) // 2
        dx = (max_w - im.shape[1]) // 2
        canvas[y+dy:y+dy+im.shape[0], x+dx:x+dx+im.shape[1]] = im

    return canvas


def apply_colormap(gray_img):
    """Apply OpenCV colormap to grayscale image (returns BGR)."""
    if len(gray_img.shape) == 2:
        return cv2.applyColorMap(gray_img, COLORMAP)
    return gray_img


def draw_stats_panel(img, model, fps, latency_ms, loss=None):
    """Overlay biological statistics on the left side."""
    h, w = img.shape[:2]
    panel_w = 220
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_w, h), (20, 20, 25), -1)

    lines = [
        "=== BIO-METRICS ===",
        f"FPS:       {fps:.1f}",
        f"Latency:   {latency_ms:.1f}ms",
        "",
        "=== POPULATION ===",
        f"Filters:   {NUM_FILTERS}",
        f"E-units:   {NUM_FILTERS//2 if USE_DALES else 'N/A'}",
        f"I-units:   {NUM_FILTERS//2 if USE_DALES else 'N/A'}",
        "",
        "=== ACTIVITY ===",
    ]

    if model.act_enc is not None:
        rates = [np.mean(model.act_enc[f]) for f in range(NUM_FILTERS)]
        avg_rate = np.mean(rates)
        sparsity = np.mean(np.array(rates) < 0.01)

        lines += [
            f"Avg rate:  {avg_rate:.3f}",
            f"Sparsity:  {sparsity*100:.1f}%",
            f"Target:    {TARGET_RATE}",
            "",
            "=== SYNAPSES ===",
            f"W_enc max: {np.max(np.abs(model.W_enc)):.3f}",
            f"W_dec max: {np.max(np.abs(model.W_dec)):.3f}",
        ]

        if loss is not None:
            lines += ["", f"Recon Loss: {loss:.4f}"]

    y = 25
    for line in lines:
        color = (180, 180, 180) if not line.startswith("=") else (100, 200, 100)
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, color, 1, cv2.LINE_AA)
        y += 22

    return np.hstack([panel, img])


def visualize_weights(model, size=48):
    """Visualize decoder weights as RGB patches."""
    patches = []
    for f in range(NUM_FILTERS):
        w = model.W_dec[f]                     # (K, K, 3)
        w_norm = normalize_for_display(w, (0, 255))  # global norm, preserves ratios
        w_big = cv2.resize(w_norm, (size, size), interpolation=cv2.INTER_NEAREST)
        patches.append(w_big)
    return tile_images(patches, cols=GRID_COLS, border=2)






def fit_to_panel(img, target_w, target_h, bg=BG_COLOR):
    """Scale image to fit inside target_w x target_h, center-pad with bg."""
    if img is None:
        return np.full((target_h, target_w, 3), bg, dtype=np.uint8)

    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = np.clip(img, 0, 255).astype(np.uint8)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    panel = np.full((target_h, target_w, 3), bg, dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    panel[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return panel


def make_label_bar(text_left, text_right, width, height=13, bg=BG_COLOR):
    bar = np.full((height, width, 3), bg, dtype=np.uint8)
    cv2.putText(bar, text_left, (5, 10), cv2.FONT_HERSHEY_SIMPLEX,
               0.35, (160, 160, 160), 1, cv2.LINE_AA)
    if text_right:
        tw, _ = cv2.getTextSize(text_right, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
        cv2.putText(bar, text_right, (width - tw - 5, 10), cv2.FONT_HERSHEY_SIMPLEX,
                   0.35, (160, 160, 160), 1, cv2.LINE_AA)
    return bar


def paste_panel(canvas, panel, x, y):
    """Paste a panel onto canvas at (x,y), clipping if needed."""
    h, w = panel.shape[:2]
    y1, y2 = max(0, y), min(canvas.shape[0], y + h)
    x1, x2 = max(0, x), min(canvas.shape[1], x + w)
    ph, pw = y2 - y1, x2 - x1
    canvas[y1:y2, x1:x2] = panel[:ph, :pw]


def live_neuron_viewer(model, train_while_viewing=False, capture_device=0):
    global COLORMAP
    colormaps = [
        cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_JET, cv2.COLORMAP_HOT,
        cv2.COLORMAP_COOL, cv2.COLORMAP_PLASMA, cv2.COLORMAP_INFERNO,
        cv2.COLORMAP_TWILIGHT, cv2.COLORMAP_TURBO
    ]
    cmap_idx = 0

    cap = cv2.VideoCapture(capture_device)
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    training = train_while_viewing
    paused = False
    show_weights = SHOW_WEIGHTS
    grid_cols = GRID_COLS
    frame_count = 0
    fps = 0.0
    t_prev = time.time()
    loss_history = []

    print("\n" + "="*55)
    print("  LIVE NEURON VIEWER  —  960x540 (50% scale)")
    print("="*55)
    print("  [q] quit  |  [t] toggle training  |  [c] cycle colormap")
    print("  [s] screenshot  |  [p] pause  |  [w] toggle weights")
    print("  [1-4] grid columns  |  [r] reset stats")
    print("="*55)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            t_start = time.time()

            # --- PREPROCESS ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
            norm = small.astype(np.float32) / 255.0

            # --- FORWARD ---
            recon = model.forward(norm)

            # --- OPTIONAL TRAIN ---
            loss = None
            if training:
                loss = 0.5 * np.sum((recon - norm) ** 2)
                if np.isfinite(loss):
                    model.backward(norm)
                    loss_history.append(loss)
                else:
                    print("  [WARNING] NaN loss, skipping update.")

            # --- BUILD MAIN CANVAS ---
            canvas = np.full((MAIN_H, MAIN_W, 3), BG_COLOR, dtype=np.uint8)

            # 1. Camera preview (small, top-left)
            cam_content = fit_to_panel(frame, PREVIEW_W, PREVIEW_H)
            cam_bar = make_label_bar("CAM", f"Trn:{training}", PREVIEW_W, PREVIEW_BAR_H)
            cam_panel = np.vstack([cam_bar, cam_content])
            paste_panel(canvas, cam_panel, 0, 0)

            # 2. Reconstruction (small, next to camera)
            recon_u8 = normalize_for_display(recon, (0, 255))
            recon_bgr = cv2.cvtColor(recon_u8, cv2.COLOR_RGB2BGR)
            recon_content = fit_to_panel(recon_bgr, PREVIEW_W, PREVIEW_H)
            recon_bar = make_label_bar("REC",
                f"L:{loss_history[-1]:.3f}" if loss_history else "L:--", PREVIEW_W, PREVIEW_BAR_H)
            recon_panel = np.vstack([recon_bar, recon_content])
            paste_panel(canvas, recon_panel, PREVIEW_W + 5, 0)

            # 3. Encoder activations (LARGE, bottom-left)
            enc_maps = [model.act_enc[f] for f in range(NUM_FILTERS)]
            enc_grid = tile_images(enc_maps, cols=grid_cols, border=1)
            enc_color = apply_colormap(enc_grid)
            enc_content = fit_to_panel(enc_color, NEURON_W, NEURON_CONTENT_H)
            enc_bar = make_label_bar("ENCODER",
                f"C:{grid_cols} D:{'ON' if USE_DALES else 'OFF'}", NEURON_W, NEURON_BAR_H)
            enc_panel = np.vstack([enc_bar, enc_content])
            paste_panel(canvas, enc_panel, 0, PREVIEW_TOTAL_H)

            # 4. Pooled features OR Decoder weights (LARGE, bottom-right)
            if show_weights:
                weight_grid = visualize_weights(model, size=40)
                weight_bgr = cv2.cvtColor(weight_grid, cv2.COLOR_RGB2BGR)
                bot_content = fit_to_panel(weight_bgr, NEURON_W, NEURON_CONTENT_H)
                bot_label = "DECODER"
            else:
                pool_maps = [model.pooled[f] for f in range(NUM_FILTERS)]
                pool_grid = tile_images(pool_maps, cols=grid_cols, border=1)
                pool_color = apply_colormap(pool_grid)
                bot_content = fit_to_panel(pool_color, NEURON_W, NEURON_CONTENT_H)
                bot_label = "POOLED"

            rates = [np.mean(model.act_enc[f]) for f in range(NUM_FILTERS)] if model.act_enc is not None else []
            avg_rate = np.mean(rates) if rates else 0.0
            bot_bar = make_label_bar(bot_label,
                f"R:{avg_rate:.3f}", NEURON_W, NEURON_BAR_H)
            bot_panel = np.vstack([bot_bar, bot_content])
            paste_panel(canvas, bot_panel, NEURON_W, PREVIEW_TOTAL_H)

            # --- TIMING & STATS ---
            latency_ms = (time.time() - t_start) * 1000
            dt = time.time() - t_prev
            t_prev = time.time()
            fps = FPS_SMOOTH * fps + (1 - FPS_SMOOTH) * (1.0 / max(dt, 0.001))

            final_view = draw_stats_panel(canvas, model, fps, latency_ms,
                                         loss=loss_history[-1] if loss_history else None)
            frame_count += 1

        # --- DISPLAY ---
        cv2.imshow("Bio-Neuron Viewer", final_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            training = not training
            print(f"  Training: {training}")
        elif key == ord('p'):
            paused = not paused
            print(f"  Paused: {paused}")
        elif key == ord('c'):
            cmap_idx = (cmap_idx + 1) % len(colormaps)
            COLORMAP = colormaps[cmap_idx]
            print(f"  Colormap: {cmap_idx}")
        elif key == ord('w'):
            show_weights = not show_weights
            print(f"  Show weights: {show_weights}")
        elif key == ord('s'):
            filename = f"neuron_viewer_{frame_count:04d}.png"
            cv2.imwrite(filename, final_view)
            print(f"  Screenshot saved: {filename}")
        elif key == ord('r'):
            loss_history.clear()
            frame_count = 0
            print("  Stats reset.")
        elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
            grid_cols = int(chr(key))
            print(f"  Grid columns: {grid_cols}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nViewer closed. Processed {frame_count} frames.")
    if loss_history:
        print(f"Final avg loss (last 100): {np.mean(loss_history[-100:]):.6f}")

def run_viewer_with_pretrained(weights_path="weights.bin", train=False):
    """Load weights and launch the viewer."""
    model = BioAutoencoder()
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"No weights found at {weights_path}, using random init.")
    live_neuron_viewer(model, train_while_viewing=train)


def run_viewer_fresh(train=True):
    """Start with random weights and optionally train live."""
    model = BioAutoencoder()
    live_neuron_viewer(model, train_while_viewing=train)


# ---------- Main Menu ----------
def main():
    print("=" * 55)
    print("  Bio-Inspired RGB Autoencoder — Pure NumPy")
    print("=" * 55)
    print(f"  Resolution : {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Filters    : {NUM_FILTERS} ({'E/I split' if USE_DALES else 'mixed'})")
    print(f"  Kernel     : {KERNEL_SIZE}x{KERNEL_SIZE}")
    print(f"  Pool       : {POOL_TYPE}")
    print(f"  LR         : {LEARNING_RATE}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Leak       : {LEAK}  |  Decay: {WEIGHT_DECAY}  |  Sparse: {SPARSITY_LAMBDA}")
    print("-" * 55)
    print("  [1] Capture dataset + train + save + test + demo")
    print("  [2] Load dataset      + train + save + test + demo")
    print("  [3] Load weights      + demo (inference only)")
    print("  [4] Load weights      + save test frame")
    print("  [5] REAL-TIME TRAINING — learn from live camera")
    print("  [6] LIVE NEURON VIEWER — visualize activations & weights")
    print("-" * 55)

    choice = input("Choice: ").strip()
    model = BioAutoencoder()

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

    elif choice == '6':
        if os.path.exists(WEIGHTS_FILE):
            load = input("Load existing weights? (y/n): ").strip().lower()
            if load == 'y':
                run_viewer_with_pretrained(WEIGHTS_FILE, train=False)
            else:
                run_viewer_fresh(train=True)
        else:
            run_viewer_fresh(train=True)

    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
