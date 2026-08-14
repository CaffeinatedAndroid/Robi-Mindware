import numpy as np
import cv2
import os
import time
import psutil

# ============ GLOBAL CONFIG ============
IMG_SIZE        = 64
NUM_FILTERS_L1  = 32
NUM_FILTERS_L2  = 64
KERNEL_SIZE     = 3
LEARNING_RATE   = 0.001
EPOCHS          = 10
CAPTURE_SECONDS = 10
DATASET_DIR     = "dataset"
WEIGHTS_FILE    = "weights.bin"
TRAIN_OUT_DIR   = "training_output"
POOL_TYPE       = "avg"

LEAK            = 0.02
WEIGHT_CLIP     = 1.5
WEIGHT_DECAY    = 5e-5
SPARSITY_LAMBDA = 5e-4
TARGET_RATE     = 0.10
HOMEOSTATIC_LR  = 0.005
USE_DALES       = True
GRAD_CLIP       = 1.0
# =======================================


def leaky_relu(x, leak=LEAK):
    return np.where(x > 0, x, leak * x)

def leaky_relu_deriv(x, leak=LEAK):
    return np.where(x > 0, 1.0, leak)


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


def upsample2x2_forward(x, target_shape):
    """Upsample a SINGLE 2D feature map."""
    h, w = x.shape
    out = np.zeros(target_shape, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            out[i*2:i*2+2, j*2:j*2+2] = x[i, j]
    return out


def upsample2x2_backward(d_upsampled, target_shape):
    """Backprop through upsample for a SINGLE 2D feature map."""
    h, w = target_shape
    dx = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            dx[i, j] = np.sum(d_upsampled[i*2:i*2+2, j*2:j*2+2])
    return dx


class BioAutoencoder:
    def __init__(self):
        F1 = NUM_FILTERS_L1
        F2 = NUM_FILTERS_L2
        C = 3
        k = KERNEL_SIZE

        fan_in = k * k * C
        self.W_enc1 = np.random.randn(F1, k, k, C).astype(np.float32) * np.sqrt(2.0 / fan_in) * 0.5
        self.b_enc1 = np.zeros(F1, dtype=np.float32)

        fan_in = k * k * F1
        self.W_enc2 = np.random.randn(F2, k, k, F1).astype(np.float32) * np.sqrt(2.0 / fan_in) * 0.5
        self.b_enc2 = np.zeros(F2, dtype=np.float32)

        fan_in = k * k * F2
        self.W_dec2 = np.random.randn(F1, k, k, F2).astype(np.float32) * np.sqrt(2.0 / fan_in) * 0.5
        self.b_dec2 = np.zeros(F1, dtype=np.float32)

        fan_in = k * k * F1
        self.W_dec1 = np.random.randn(C, k, k, F1).astype(np.float32) * np.sqrt(2.0 / fan_in) * 0.5
        self.b_dec1 = np.full(C, 0.5, dtype=np.float32)

        if USE_DALES:
            self.E_mask1 = np.ones((F1, 1, 1, 1), dtype=np.float32)
            self.I_mask1 = np.ones_like(self.E_mask1)
            self.E_mask1[F1//2:] = 0
            self.I_mask1[:F1//2] = 0
            self.W_enc1 = np.abs(self.W_enc1) * self.E_mask1 - np.abs(self.W_enc1) * self.I_mask1

            self.E_mask2 = np.ones((F2, 1, 1, 1), dtype=np.float32)
            self.I_mask2 = np.ones_like(self.E_mask2)
            self.E_mask2[F2//2:] = 0
            self.I_mask2[:F2//2] = 0
            self.W_enc2 = np.abs(self.W_enc2) * self.E_mask2 - np.abs(self.W_enc2) * self.I_mask2

        self.x = None
        self.conv1 = None
        self.act1 = None
        self.p1 = None
        self.pool_masks1 = None
        self.conv2 = None
        self.act2 = None
        self.p2 = None
        self.pool_masks2 = None
        self.u2 = None
        self.u2_skip = None
        self.dec2 = None
        self.u1 = None
        self.u1_skip = None
        self.output = None

    def _clip_weights(self):
        np.clip(self.W_enc1, -WEIGHT_CLIP, WEIGHT_CLIP, out=self.W_enc1)
        np.clip(self.W_enc2, -WEIGHT_CLIP, WEIGHT_CLIP, out=self.W_enc2)
        np.clip(self.W_dec2, -WEIGHT_CLIP, WEIGHT_CLIP, out=self.W_dec2)
        np.clip(self.W_dec1, -WEIGHT_CLIP, WEIGHT_CLIP, out=self.W_dec1)

    def _enforce_dales(self):
        if USE_DALES:
            self.W_enc1 = np.abs(self.W_enc1) * self.E_mask1 - np.abs(self.W_enc1) * self.I_mask1
            self.W_enc2 = np.abs(self.W_enc2) * self.E_mask2 - np.abs(self.W_enc2) * self.I_mask2

    def _clip_grad(self, grad):
        return np.clip(grad, -GRAD_CLIP, GRAD_CLIP)

    def _check_nan(self, name):
        for tensor, label in [(self.W_enc1, "W_enc1"), (self.W_enc2, "W_enc2"),
                              (self.W_dec2, "W_dec2"), (self.W_dec1, "W_dec1"),
                              (self.b_enc1, "b_enc1"), (self.b_enc2, "b_enc2"),
                              (self.b_dec2, "b_dec2"), (self.b_dec1, "b_dec1")]:
            if not np.isfinite(tensor).all():
                print(f"  [WARNING] NaN/Inf in {label} after {name}!")
                return True
        return False

    def forward(self, x):
        self.x = x.astype(np.float32)
        H, W, C = self.x.shape
        F1 = NUM_FILTERS_L1
        F2 = NUM_FILTERS_L2

        # === ENCODER LAYER 1 ===
        self.conv1 = np.zeros((F1, H, W), dtype=np.float32)
        for f in range(F1):
            for c in range(C):
                self.conv1[f] += conv2d(self.x[:, :, c], self.W_enc1[f, :, :, c], padding=1)
            self.conv1[f] += self.b_enc1[f]
        self.act1 = leaky_relu(self.conv1)

        self.p1 = np.zeros((F1, H//2, W//2), dtype=np.float32)
        self.pool_masks1 = []
        for f in range(F1):
            if POOL_TYPE == "max":
                p, m = maxpool2x2_forward(self.act1[f])
                self.p1[f] = p
                self.pool_masks1.append(m)
            else:
                self.p1[f] = avgpool2x2_forward(self.act1[f])
                self.pool_masks1.append(None)

        # === ENCODER LAYER 2 ===
        self.conv2 = np.zeros((F2, H//2, W//2), dtype=np.float32)
        for f in range(F2):
            for c in range(F1):
                self.conv2[f] += conv2d(self.p1[c], self.W_enc2[f, :, :, c], padding=1)
            self.conv2[f] += self.b_enc2[f]
        self.act2 = leaky_relu(self.conv2)

        self.p2 = np.zeros((F2, H//4, W//4), dtype=np.float32)
        self.pool_masks2 = []
        for f in range(F2):
            if POOL_TYPE == "max":
                p, m = maxpool2x2_forward(self.act2[f])
                self.p2[f] = p
                self.pool_masks2.append(m)
            else:
                self.p2[f] = avgpool2x2_forward(self.act2[f])
                self.pool_masks2.append(None)

        # === DECODER LAYER 2 ===
        # FIX: loop over filters for upsample (was passing 3D array to 2D function)
        self.u2 = np.zeros((F2, H//2, W//2), dtype=np.float32)
        for f in range(F2):
            self.u2[f] = upsample2x2_forward(self.p2[f], (H//2, W//2))
        self.u2_skip = self.u2 + self.act2

        self.dec2 = np.zeros((F1, H//2, W//2), dtype=np.float32)
        for f in range(F1):
            for c in range(F2):
                self.dec2[f] += conv2d(self.u2_skip[c], self.W_dec2[f, :, :, c], padding=1)
            self.dec2[f] += self.b_dec2[f]

        # === DECODER LAYER 1 ===
        # FIX: loop over filters for upsample
        self.u1 = np.zeros((F1, H, W), dtype=np.float32)
        for f in range(F1):
            self.u1[f] = upsample2x2_forward(self.dec2[f], (H, W))
        self.u1_skip = self.u1 + self.act1

        self.output = np.zeros((H, W, C), dtype=np.float32)
        for c in range(C):
            for f in range(F1):
                self.output[:, :, c] += conv2d(self.u1_skip[f], self.W_dec1[c, :, :, f], padding=1)
            self.output[:, :, c] += self.b_dec1[c]

        self.output = np.clip(self.output, -2.0, 2.0)
        return self.output

    def backward(self, target):
        H, W, C = self.x.shape
        F1 = NUM_FILTERS_L1
        F2 = NUM_FILTERS_L2

        d_pre = self.output - target

        self.b_dec1 -= LEARNING_RATE * np.sum(d_pre, axis=(0, 1))
        np.clip(self.b_dec1, -2.0, 2.0, out=self.b_dec1)

        dW_dec1 = np.zeros_like(self.W_dec1)
        d_u1_skip = np.zeros((F1, H, W), dtype=np.float32)

        for c in range(C):
            for f in range(F1):
                dW_dec1[c, :, :, f] += conv2d_grad_kernel(
                    self.u1_skip[f], d_pre[:, :, c], (KERNEL_SIZE, KERNEL_SIZE)
                )
                d_u1_skip[f] += conv2d_grad_input(
                    d_pre[:, :, c], self.W_dec1[c, :, :, f], (H, W)
                )

        dW_dec1 = self._clip_grad(dW_dec1)
        self.W_dec1 -= LEARNING_RATE * dW_dec1

        d_u1 = d_u1_skip
        d_act1 = d_u1_skip.copy()

        # FIX: loop over filters for upsample backward
        d_dec2 = np.zeros((F1, H//2, W//2), dtype=np.float32)
        for f in range(F1):
            d_dec2[f] = upsample2x2_backward(d_u1[f], (H//2, W//2))

        self.b_dec2 -= LEARNING_RATE * np.sum(d_dec2, axis=(1, 2))
        np.clip(self.b_dec2, -2.0, 2.0, out=self.b_dec2)

        dW_dec2 = np.zeros_like(self.W_dec2)
        d_u2_skip = np.zeros((F2, H//2, W//2), dtype=np.float32)

        for f in range(F1):
            for c in range(F2):
                dW_dec2[f, :, :, c] += conv2d_grad_kernel(
                    self.u2_skip[c], d_dec2[f], (KERNEL_SIZE, KERNEL_SIZE)
                )
                d_u2_skip[c] += conv2d_grad_input(
                    d_dec2[f], self.W_dec2[f, :, :, c], (H//2, W//2)
                )

        dW_dec2 = self._clip_grad(dW_dec2)
        self.W_dec2 -= LEARNING_RATE * dW_dec2

        d_u2 = d_u2_skip
        d_act2 = d_u2_skip.copy()

        # FIX: loop over filters for upsample backward
        d_p2 = np.zeros((F2, H//4, W//4), dtype=np.float32)
        for f in range(F2):
            d_p2[f] = upsample2x2_backward(d_u2[f], (H//4, W//4))

        for f in range(F2):
            if POOL_TYPE == "max":
                d_act2[f] += maxpool2x2_backward(d_p2[f], self.pool_masks2[f])
            else:
                d_act2[f] += avgpool2x2_backward(d_p2[f], (H//2, W//2))

        d_act2 += SPARSITY_LAMBDA * np.sign(self.act2)
        d_conv2 = d_act2 * leaky_relu_deriv(self.conv2)

        for f in range(F2):
            error = np.mean(self.act2[f]) - TARGET_RATE
            self.b_enc2[f] -= HOMEOSTATIC_LR * error
        np.clip(self.b_enc2, -2.0, 2.0, out=self.b_enc2)

        dW_enc2 = np.zeros_like(self.W_enc2)
        d_p1 = np.zeros_like(self.p1)

        for f in range(F2):
            for c in range(F1):
                dW_enc2[f, :, :, c] += conv2d_grad_kernel(
                    self.p1[c], d_conv2[f], (KERNEL_SIZE, KERNEL_SIZE)
                )
                d_p1[c] += conv2d_grad_input(
                    d_conv2[f], self.W_enc2[f, :, :, c], (H//2, W//2)
                )

        dW_enc2 = self._clip_grad(dW_enc2)
        self.W_enc2 -= LEARNING_RATE * dW_enc2

        for f in range(F1):
            if POOL_TYPE == "max":
                d_act1[f] += maxpool2x2_backward(d_p1[f], self.pool_masks1[f])
            else:
                d_act1[f] += avgpool2x2_backward(d_p1[f], (H, W))

        d_act1 += SPARSITY_LAMBDA * np.sign(self.act1)
        d_conv1 = d_act1 * leaky_relu_deriv(self.conv1)

        for f in range(F1):
            error = np.mean(self.act1[f]) - TARGET_RATE
            self.b_enc1[f] -= HOMEOSTATIC_LR * error
        np.clip(self.b_enc1, -2.0, 2.0, out=self.b_enc1)

        dW_enc1 = np.zeros_like(self.W_enc1)
        for f in range(F1):
            for c in range(C):
                dW_enc1[f, :, :, c] += conv2d_grad_kernel(
                    self.x[:, :, c], d_conv1[f], (KERNEL_SIZE, KERNEL_SIZE)
                )

        dW_enc1 = self._clip_grad(dW_enc1)
        self.W_enc1 -= LEARNING_RATE * dW_enc1

        self.W_enc1 -= LEARNING_RATE * WEIGHT_DECAY * self.W_enc1
        self.W_enc2 -= LEARNING_RATE * WEIGHT_DECAY * self.W_enc2
        self.W_dec2 -= LEARNING_RATE * WEIGHT_DECAY * self.W_dec2
        self.W_dec1 -= LEARNING_RATE * WEIGHT_DECAY * self.W_dec1

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
            self.W_enc1.flatten(), self.W_enc2.flatten(),
            self.W_dec2.flatten(), self.W_dec1.flatten(),
            self.b_enc1.flatten(), self.b_enc2.flatten(),
            self.b_dec2.flatten(), self.b_dec1.flatten()
        ]).astype(np.float32)
        flat.tofile(path)
        print(f"Saved {len(flat)} floats -> {path}")

    def load_weights(self, path):
        flat = np.fromfile(path, dtype=np.float32)
        idx = 0
        F1 = NUM_FILTERS_L1
        F2 = NUM_FILTERS_L2
        C = 3
        k = KERNEL_SIZE

        n = F1 * k * k * C
        self.W_enc1 = flat[idx:idx+n].reshape(F1, k, k, C); idx += n
        n = F2 * k * k * F1
        self.W_enc2 = flat[idx:idx+n].reshape(F2, k, k, F1); idx += n
        n = F1 * k * k * F2
        self.W_dec2 = flat[idx:idx+n].reshape(F1, k, k, F2); idx += n
        n = C * k * k * F1
        self.W_dec1 = flat[idx:idx+n].reshape(C, k, k, F1); idx += n
        self.b_enc1 = flat[idx:idx+F1]; idx += F1
        self.b_enc2 = flat[idx:idx+F2]; idx += F2
        self.b_dec2 = flat[idx:idx+F1]; idx += F1
        self.b_dec1 = flat[idx:idx+C]
        print(f"Loaded weights from {path}")


def capture_dataset():
    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"Capturing for {CAPTURE_SECONDS}s — press 'q' to stop early.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera failed to open.")
        cap.release()
        return []
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
    if test_frame is None:
        print("[ERROR] No test frame available.")
        return model
    if len(data) == 0:
        print("[ERROR] No training data.")
        return model

    save_training_image(model, test_frame, 0, TRAIN_OUT_DIR)

    print("=== TRAINING START ===")
    print(f"Epochs: {EPOCHS} | Images: {len(data)} | L1: {NUM_FILTERS_L1} | L2: {NUM_FILTERS_L2} | Pool: {POOL_TYPE} | LR: {LEARNING_RATE}")
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
                print(f"  [WARNING] NaN loss at step {i}, skipping.")
                break
            epoch_loss += loss

        avg_loss = epoch_loss / len(data)
        if not np.isfinite(avg_loss):
            print("  [ERROR] Epoch produced NaN. Lower LEARNING_RATE.")
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
    if not cap.isOpened():
        print("[ERROR] Camera not available.")
        return

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
                print("  [WARNING] NaN loss, skipping frame.")

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
    if not cap.isOpened():
        print("[ERROR] Camera not available.")
        return
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
    if not cap.isOpened():
        print("[ERROR] Camera not available.")
        return
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
    if not cap.isOpened():
        print("[ERROR] Camera not available for test frame.")
        cap.release()
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Camera read failed.")
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    return small.astype(np.float32) / 255.0


def main():
    print("=" * 60)
    print("  Deep Bio-U-Net Autoencoder — Pure NumPy")
    print("=" * 60)
    print(f"  Resolution : {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Enc1       : {NUM_FILTERS_L1} filters (32x32)")
    print(f"  Enc2       : {NUM_FILTERS_L2} filters (8x8 latent)")
    print(f"  Skips      : 2 (U-Net style)")
    print(f"  Kernel     : {KERNEL_SIZE}x{KERNEL_SIZE}")
    print(f"  Pool       : {POOL_TYPE}")
    print(f"  LR         : {LEARNING_RATE}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Leak       : {LEAK}  |  Decay: {WEIGHT_DECAY}  |  Sparse: {SPARSITY_LAMBDA}")
    print("-" * 60)
    print("  [1] Capture dataset + train + save + test + demo")
    print("  [2] Load dataset      + train + save + test + demo")
    print("  [3] Load weights      + demo (inference only)")
    print("  [4] Load weights      + save test frame")
    print("  [5] REAL-TIME TRAINING — learn from live camera")
    print("-" * 60)

    choice = input("Choice: ").strip()
    model = BioAutoencoder()

    if choice == '1':
        data = capture_dataset()
        if len(data) == 0:
            print("[ABORT] No frames captured.")
            return
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
        if len(data) == 0:
            print("[ABORT] Dataset folder is empty.")
            return
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
