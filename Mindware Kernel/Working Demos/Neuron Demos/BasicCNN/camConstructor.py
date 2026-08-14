import numpy as np
import cv2
import os
import time
import psutil

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
IMG_SIZE        = 32
NUM_FILTERS     = 16
KERNEL_SIZE     = 3
LEARNING_RATE   = 0.0004       # Reduced for stability on small/dark datasets
EPOCHS          = 100
CAPTURE_SECONDS = 20             # More data prevents overfitting to noise
DATASET_DIR     = "dataset"
WEIGHTS_FILE    = "weights.bin"
TRAIN_OUT_DIR   = "training_output"
POOL_TYPE       = "avg"

# Gamma correction: lifts dark camera feeds into a visible, trainable range
AUTO_GAMMA_ENABLED = True
GAMMA_TARGET_MEAN  = 0.45


# ============================================================================
# CAMERA & IMAGE FIXES
# ============================================================================

def open_camera():
    """
    Opens the camera with auto-exposure enabled and discards warm-up frames.
    Many webcams default to manual minimum exposure, producing black images.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera index 0.")
        return None

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    cap.set(cv2.CAP_PROP_EXPOSURE, 0.5)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("[CAMERA] Warming up sensor (discarding 10 frames)...")
    for i in range(10):
        ret, _ = cap.read()
        if not ret:
            print(f"[CAMERA] Warning: warm-up frame {i} failed.")

    return cap


def debug_frame(frame, label="Frame"):
    """
    Prints per-channel brightness stats so you can verify the image isn't black.
    """
    if frame.dtype == np.uint8:
        f = frame.astype(np.float32) / 255.0
    else:
        f = frame

    if f.ndim == 3 and f.shape[2] == 3:
        means = [f[:, :, c].mean() for c in range(3)]
        print(f"[DEBUG] {label} | Shape: {f.shape} | Means [R,G,B]: [{means[0]:.3f}, {means[1]:.3f}, {means[2]:.3f}]")
    else:
        print(f"[DEBUG] {label} | Shape: {f.shape} | Mean: {f.mean():.3f} | Max: {f.max():.3f}")


def auto_gamma(image_rgb, target_mean=GAMMA_TARGET_MEAN):
    """
    Non-linear gamma correction that lifts dark shadows without clipping highlights.
    If the image is already bright enough, it is returned unchanged.
    """
    f = image_rgb.astype(np.float32) / 255.0
    mean = f.mean()

    if mean >= target_mean or mean < 0.001:
        return image_rgb

    gamma = np.log(target_mean) / np.log(mean)
    corrected = np.power(f, gamma) * 255.0
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    print(f"[GAMMA] mean {mean:.3f} -> {corrected.mean()/255.0:.3f} (gamma={gamma:.3f})")
    return corrected


# ============================================================================
# VECTORIZED 2D CONVOLUTION
# ============================================================================
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


# ---------- Activations ----------
def relu(x):
    return np.maximum(0.0, x)

def relu_deriv(x):
    return (x > 0).astype(np.float32)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv_from_output(out):
    return out * (1.0 - out)


# ---------- Pooling ----------
def maxpool2x2_forward(x):
    H, W = x.shape
    H -= H % 2
    W -= W % 2
    x = x[:H, :W]

    out_h, out_w = H // 2, W // 2
    pooled = np.zeros((out_h, out_w), dtype=np.float32)
    mask   = np.zeros((H, W), dtype=np.float32)

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
            grad = d_pooled[i, j]
            block = mask[i*2:i*2+2, j*2:j*2+2]
            dx[i*2:i*2+2, j*2:j*2+2] = block * grad
    return dx


def avgpool2x2_forward(x):
    H, W = x.shape
    H -= H % 2
    W -= W % 2
    x = x[:H, :W]

    out_h, out_w = H // 2, W // 2
    pooled = np.zeros((out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            patch = x[i*2:i*2+2, j*2:j*2+2]
            pooled[i, j] = np.mean(patch)

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
class SimpleAutoencoder:
    def __init__(self):
        # Per-channel encoder weights: each filter learns separate R, G, B detectors
        self.W_enc = np.random.randn(NUM_FILTERS, KERNEL_SIZE, KERNEL_SIZE, 3).astype(np.float32)
        fan_in = KERNEL_SIZE * KERNEL_SIZE * 3
        self.W_enc *= np.sqrt(2.0 / fan_in)

        # Decoder: per-filter, per-output-channel kernels
        fan_in_dec = KERNEL_SIZE * KERNEL_SIZE
        self.W_dec = np.random.randn(NUM_FILTERS, KERNEL_SIZE, KERNEL_SIZE, 3).astype(np.float32)
        self.W_dec *= np.sqrt(2.0 / fan_in_dec)

        # Caches
        self.x          = None
        self.conv_enc   = None
        self.relu_enc   = None
        self.pooled     = None
        self.pool_masks = None
        self.upsampled  = None
        self.conv_dec   = None
        self.output     = None

    def forward(self, x):
        self.x = x.astype(np.float32)
        H, W, C = self.x.shape

        self.conv_enc = np.zeros((NUM_FILTERS, H, W), dtype=np.float32)
        for f in range(NUM_FILTERS):
            for c in range(C):
                self.conv_enc[f] += conv2d(self.x[:, :, c], self.W_enc[f, :, :, c], padding=1)

        self.relu_enc = relu(self.conv_enc)

        self.pooled = np.zeros((NUM_FILTERS, H//2, W//2), dtype=np.float32)
        self.pool_masks = []

        for f in range(NUM_FILTERS):
            if POOL_TYPE == "max":
                p, m = maxpool2x2_forward(self.relu_enc[f])
                self.pooled[f] = p
                self.pool_masks.append(m)
            else:
                self.pooled[f] = avgpool2x2_forward(self.relu_enc[f])
                self.pool_masks.append(None)

        self.upsampled = np.zeros((NUM_FILTERS, H, W), dtype=np.float32)
        for f in range(NUM_FILTERS):
            self.upsampled[f] = upsample2x2(self.pooled[f], (H, W))

        self.conv_dec = np.zeros((H, W, C), dtype=np.float32)
        for f in range(NUM_FILTERS):
            for c in range(C):
                self.conv_dec[:, :, c] += conv2d(
                    self.upsampled[f],
                    self.W_dec[f, :, :, c],
                    padding=1
                )

        self.output = sigmoid(self.conv_dec)
        return self.output

    def backward(self, target):
        C = 3
        d_pre = (self.output - target) * sigmoid_deriv_from_output(self.output)

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

        self.W_dec -= LEARNING_RATE * dW_dec

        d_pooled = np.zeros_like(self.pooled)
        for f in range(NUM_FILTERS):
            h_p, w_p = self.pooled[f].shape
            for i in range(h_p):
                for j in range(w_p):
                    block = d_upsampled[f, i*2:i*2+2, j*2:j*2+2]
                    d_pooled[f, i, j] = np.sum(block)

        d_relu = np.zeros_like(self.relu_enc)
        for f in range(NUM_FILTERS):
            if POOL_TYPE == "max":
                d_relu[f] = maxpool2x2_backward(d_pooled[f], self.pool_masks[f])
            else:
                d_relu[f] = avgpool2x2_backward(d_pooled[f], self.relu_enc[f].shape)

        d_conv_enc = d_relu * relu_deriv(self.conv_enc)

        dW_enc = np.zeros_like(self.W_enc)
        for f in range(NUM_FILTERS):
            for c in range(C):
                dW_enc[f, :, :, c] += conv2d_grad_kernel(
                    self.x[:, :, c], d_conv_enc[f], (KERNEL_SIZE, KERNEL_SIZE)
                )

        self.W_enc -= LEARNING_RATE * dW_enc

    def train_step(self, x):
        out = self.forward(x)
        loss = 0.5 * np.sum((out - x) ** 2)
        self.backward(x)
        return loss

    def save_weights(self, path):
        flat = np.concatenate([
            self.W_enc.flatten(),
            self.W_dec.flatten()
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
        print(f"Loaded weights from {path}")


# ---------- Dataset ----------
def capture_dataset():
    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"Capturing for {CAPTURE_SECONDS}s — press 'q' to stop early.")

    cap = open_camera()
    if cap is None:
        print("[FATAL] Camera could not be opened. Exiting.")
        return []

    frames = []
    t0 = time.time()

    while time.time() - t0 < CAPTURE_SECONDS:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for processing
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply gamma correction so the image is bright enough to train on
        if AUTO_GAMMA_ENABLED:
            rgb = auto_gamma(rgb, target_mean=GAMMA_TARGET_MEAN)

        # --- PREVIEW: show a regular, visible picture ---
        # Convert back to BGR for OpenCV display so colors look natural
        preview = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("Camera Preview — press Q to stop", preview)

        # Resize and normalize for the dataset
        small = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        norm = small.astype(np.float32) / 255.0
        frames.append(norm)

        # Debug first frame to confirm brightness
        if len(frames) == 1:
            debug_frame(norm, "First training frame (normalized)")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[CAPTURE] Stopped early by user.")
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
    if len(frames) > 0:
        debug_frame(frames[0], "First loaded frame")
    return frames


# ---------- Training & Demo ----------
def save_training_image(model, test_frame, epoch, folder):
    os.makedirs(folder, exist_ok=True)
    recon = model.forward(test_frame)
    recon_u8 = (np.clip(recon, 0, 1) * 255).astype(np.uint8)
    orig_u8 = (np.clip(test_frame, 0, 1) * 255).astype(np.uint8)
    combined = np.hstack([orig_u8, recon_u8])
    path = os.path.join(folder, f"epoch_{epoch:03d}.png")
    cv2.imwrite(path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    return path


def train(model, data, test_frame):
    save_training_image(model, test_frame, 0, TRAIN_OUT_DIR)

    print("=== TRAINING START ===")
    print(f"Epochs: {EPOCHS} | Images: {len(data)} | Filters: {NUM_FILTERS} | Pool: {POOL_TYPE} | LR: {LEARNING_RATE}")
    print(f"{'Epoch':<8} | {'Loss':<10} | {'Time':<8} | {'CPU %':<8} | {'RAM %':<8} | {'ETA':<8}")
    print("-" * 65)

    start_time = time.time()
    psutil.cpu_percent(interval=None)

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        indices = np.random.permutation(len(data))
        epoch_loss = 0.0

        for i in indices:
            epoch_loss += model.train_step(data[i])

        avg_loss = epoch_loss / len(data)
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


def demo(model):
    print("Demo running — press 'q' to quit.")

    cap = open_camera()
    if cap is None:
        print("[FATAL] Camera could not be opened for demo.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process through the same pipeline as training
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if AUTO_GAMMA_ENABLED:
            rgb = auto_gamma(rgb, target_mean=GAMMA_TARGET_MEAN)

        # --- PREVIEW: show regular picture on the left ---
        # Convert gamma-corrected RGB back to BGR for natural display
        preview_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        left = cv2.resize(preview_bgr, (300, 300), interpolation=cv2.INTER_NEAREST)

        # Network input
        small = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        norm = small.astype(np.float32) / 255.0

        recon = model.forward(norm)
        recon_u8 = (np.clip(recon, 0, 1) * 255).astype(np.uint8)
        recon_bgr = cv2.cvtColor(recon_u8, cv2.COLOR_RGB2BGR)
        right = cv2.resize(recon_bgr, (300, 300), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("Original | Reconstructed", np.hstack([left, right]))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def save_test_frame(model):
    cap = open_camera()
    if cap is None:
        print("[FATAL] Camera could not be opened.")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Camera failed.")
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if AUTO_GAMMA_ENABLED:
        rgb = auto_gamma(rgb, target_mean=GAMMA_TARGET_MEAN)

    small = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    norm = small.astype(np.float32) / 255.0
    recon = model.forward(norm)
    recon_u8 = (np.clip(recon, 0, 1) * 255).astype(np.uint8)

    combined = np.hstack([small, recon_u8])
    cv2.imwrite("test_frame.png", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print("Saved test_frame.png")

    debug_frame(norm, "Test frame original")
    debug_frame(recon, "Test frame reconstruction")


def get_test_frame():
    cap = open_camera()
    if cap is None:
        print("Camera failed — using first dataset frame.")
        return None

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Camera failed — using first dataset frame.")
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if AUTO_GAMMA_ENABLED:
        rgb = auto_gamma(rgb, target_mean=GAMMA_TARGET_MEAN)

    small = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    norm = small.astype(np.float32) / 255.0
    debug_frame(norm, "Test frame")
    return norm


# ---------- Main Menu ----------
def main():
    print("=" * 50)
    print("  Simple RGB Autoencoder — Pure NumPy")
    print("=" * 50)
    print(f"  Resolution : {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Filters    : {NUM_FILTERS}")
    print(f"  Kernel     : {KERNEL_SIZE}x{KERNEL_SIZE}")
    print(f"  Pool       : {POOL_TYPE}")
    print(f"  LR         : {LEARNING_RATE}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Gamma      : {'ON' if AUTO_GAMMA_ENABLED else 'OFF'} (target_mean={GAMMA_TARGET_MEAN})")
    print("-" * 50)
    print("  [1] Capture dataset + train + save weights + test frame + demo")
    print("  [2] Load dataset      + train + save weights + test frame + demo")
    print("  [3] Load weights      + demo")
    print("  [4] Load weights      + save test frame")
    print("-" * 50)

    choice = input("Choice: ").strip()
    model = SimpleAutoencoder()

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

    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
