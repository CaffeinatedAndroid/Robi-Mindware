import numpy as np
import cv2
import os
import time
import psutil

# ============ GLOBAL CONFIG ============
IMG_SIZE        = 32
NUM_FILTERS     = 16
KERNEL_SIZE     = 3
LEARNING_RATE   = 0.001       # LOWERED: biological features need gentler steps
EPOCHS          = 20
CAPTURE_SECONDS = 10
DATASET_DIR     = "dataset"
WEIGHTS_FILE    = "weights.bin"
TRAIN_OUT_DIR   = "training_output"
POOL_TYPE       = "avg"

# --- Biological neuron settings ---
LEAK            = 0.02
WEIGHT_CLIP     = 1.5         # TIGHTENED: smaller max synaptic strength
WEIGHT_DECAY    = 5e-5        # LOWERED: gentler forgetting
SPARSITY_LAMBDA = 5e-4        # LOWERED: less aggressive sparsity
TARGET_RATE     = 0.10        # LOWERED: easier target to hit
HOMEOSTATIC_LR  = 0.005       # LOWERED: slower bias adaptation
USE_DALES       = True
GRAD_CLIP       = 1.0         # NEW: max gradient norm before update
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
        # More conservative init: divide by 2 to prevent early explosion
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
        """
        Gradient clipping by value.  If any element exceeds GRAD_CLIP in
        magnitude, clamp it.  Prevents exploding gradients.
        """
        return np.clip(grad, -GRAD_CLIP, GRAD_CLIP)

    def _check_nan(self, name):
        """Detect NaN/Inf and print diagnostic info."""
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

        # Soft clamp: prevent runaway output from destroying gradients
        self.output = np.clip(self.output, -2.0, 2.0)

        return self.output

    def backward(self, target):
        C = 3
        d_pre = self.output - target

        # Decoder bias
        self.b_dec -= LEARNING_RATE * np.sum(d_pre, axis=(0, 1))
        np.clip(self.b_dec, -2.0, 2.0, out=self.b_dec)

        # Decoder weights
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

        # Upsample backprop
        d_pooled = np.zeros_like(self.pooled)
        for f in range(NUM_FILTERS):
            h_p, w_p = self.pooled[f].shape
            for i in range(h_p):
                for j in range(w_p):
                    d_pooled[f, i, j] = np.sum(d_upsampled[f, i*2:i*2+2, j*2:j*2+2])

        # Pool backprop
        d_act = np.zeros_like(self.act_enc)
        for f in range(NUM_FILTERS):
            if POOL_TYPE == "max":
                d_act[f] = maxpool2x2_backward(d_pooled[f], self.pool_masks[f])
            else:
                d_act[f] = avgpool2x2_backward(d_pooled[f], self.act_enc[f].shape)

        # Leaky ReLU backprop
        d_conv_enc = d_act * leaky_relu_deriv(self.conv_enc)

        # Homeostatic plasticity
        for f in range(NUM_FILTERS):
            mean_rate = np.mean(self.act_enc[f])
            error = mean_rate - TARGET_RATE
            self.b_enc[f] -= HOMEOSTATIC_LR * error
        np.clip(self.b_enc, -2.0, 2.0, out=self.b_enc)

        # Sparsity penalty
        d_conv_enc += SPARSITY_LAMBDA * np.sign(self.act_enc)

        # Encoder weights
        dW_enc = np.zeros_like(self.W_enc)
        for f in range(NUM_FILTERS):
            for c in range(C):
                dW_enc[f, :, :, c] += conv2d_grad_kernel(
                    self.x[:, :, c], d_conv_enc[f], (KERNEL_SIZE, KERNEL_SIZE)
                )

        dW_enc = self._clip_grad(dW_enc)
        self.W_enc -= LEARNING_RATE * dW_enc

        # Biological constraints
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

    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
