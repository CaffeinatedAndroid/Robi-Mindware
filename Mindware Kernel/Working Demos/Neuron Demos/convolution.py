import numpy as np
import cv2
import time
import os





import cv2
import os
import numpy as np

def create_and_save_dataset(base_folder="dataset", images_per_class=200, img_size=(32, 32)):
    cap = cv2.VideoCapture(0)
    labels = ["no_hand", "hand"]

    # Create base folder if it doesn't exist
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    print("=== DATASET CREATION ===")
    print(f"Saving {images_per_class} images per class to '{base_folder}/'")

    for label in labels:
        # Create class subfolder
        class_folder = os.path.join(base_folder, label)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

        print(f"\nReady to capture '{label}'? Point camera and press 's' to start...")
        if input() != 's':
            print(f"Skipped '{label}'.")
            continue

        count = 0
        print(f"Capturing {images_per_class} images for '{label}'... Move your hand around!")
        print("Press 'q' anytime to stop early.")

        while count < images_per_class:
            ret, frame = cap.read()
            if not ret: break

            # Preprocess
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, img_size)

            # Save image to disk (filename: 0.png, 1.png, ...)
            filename = os.path.join(class_folder, f"{count}.png")
            cv2.imwrite(filename, resized)

            # Show feedback
            cv2.imshow("Capture", resized)
            count += 1

            # Press 'q' to break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print(f"Saved {count} images to '{class_folder}'")

    cap.release()
    cv2.destroyAllWindows()
    print("\nDataset creation complete!")

# Run this once to generate your data
# create_and_save_dataset()



def load_dataset_from_folder(base_folder="dataset", img_size=(32, 32)):
    X_data = []
    y_data = []
    label_map = {}

    # Get list of class folders (sorted for consistent labeling)
    classes = sorted([d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))])

    print(f"Loading dataset from '{base_folder}'...")
    print(f"Found classes: {classes}")

    for label_id, class_name in enumerate(classes):
        class_path = os.path.join(base_folder, class_name)
        label_map[class_name] = label_id

        # Iterate over all images in the class folder
        for filename in os.listdir(class_path):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                img_path = os.path.join(class_path, filename)

                # Load image
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue

                # Ensure size matches (in case saved images vary)
                img = cv2.resize(img, img_size)

                # Normalize
                img = img / 255.0

                X_data.append(img.flatten()) # Flatten for MLP/CNN input
                y_data.append(label_id)

    X_data = np.array(X_data)
    y_data = np.array(y_data)


    print(f"Loaded {len(X_data)} images. Shape: {X_data.shape}")
    return X_data, y_data, len(classes)

# --- Usage Example ---
# 1. Generate data (Run create_and_save_dataset() first)
# 2. Load data
# X_train, y_train, num_classes = load_dataset_from_folder()

# # 3. One-hot encode labels for your CNN
# def one_hot_encode(labels, num_classes):
#      one_hot = np.zeros((labels.size, num_classes))
#      one_hot[np.arange(labels.size), labels] = 1
#      return one_hot
# y_train_one_hot = one_hot_encode(y_train, num_classes)




def convolve2d(image, kernel, stride=1, padding=0, mode='valid'):
    # Apply padding if needed
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')

    h, w = image.shape
    kh, kw = kernel.shape

    if mode == 'valid':
        out_h = (h - kh) // stride + 1
        out_w = (w - kw) // stride + 1
    else: # 'full' mode for backprop
        out_h = (h + kh) - 1
        out_w = (w + kw) - 1

    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            if mode == 'valid':
                region = image[i*stride : i*stride+kh, j*stride : j*stride+kw]
                output[i, j] = np.sum(region * kernel)
            else: # Full convolution logic (sliding kernel over padded error)
                # Define region in image that overlaps with kernel
                k_i_start = max(0, kh - 1 - i)
                k_i_end = min(kh, h - i + kh - 1)
                k_j_start = max(0, kw - 1 - j)
                k_j_end = min(kw, w - j + kw - 1)

                i_start = i - (kh - 1) + k_i_start
                j_start = j - (kw - 1) + k_j_start

                region = image[i_start:i_start+(k_i_end-k_i_start),
                               j_start:j_start+(k_j_end-k_j_start)]
                kernel_part = kernel[k_i_start:k_i_end, k_j_start:k_j_end]

                output[i, j] = np.sum(region * kernel_part)
    return output

class SimpleCNN:
    def __init__(self, input_features, num_classes):
        # Conv Layer
        self.filters = np.random.randn(4, 3, 3) * 0.1
        self.bias = np.zeros(4)
        self.lr = 0.01

        # Dense Layer (New)
        self.W_dense = np.random.randn(input_features, num_classes) * 0.1
        self.b_dense = np.zeros(num_classes)

        # Store for backprop if needed
        self.num_classes = num_classes
        self.input_features = input_features

    def forward(self, x):
        self.input = x
        self.feature_maps = []
        self.conv_outputs = []

        # Convolution + ReLU
        for f in self.filters:
            conv = convolve2d(x, f, stride=1, padding=1)
            self.conv_outputs.append(conv)
            self.feature_maps.append(np.maximum(0, conv))

        self.feature_maps = np.array(self.feature_maps)

        # DEBUG: Check shape here
        # print(f"Feature Maps Shape: {self.feature_maps.shape}")
        # Expected: (4, 32, 32) for 32x32 input

        if self.feature_maps.size == 0:
            raise ValueError("Feature maps are empty. Check convolution output.")

        # Pooling Step
        h, w = self.feature_maps.shape[1], self.feature_maps.shape[2]

        # Ensure dimensions are even for 2x2 pooling
        if h % 2 != 0 or w % 2 != 0:
            # Crop or pad to make even
            h = h - (h % 2)
            w = w - (w % 2)
            self.feature_maps = self.feature_maps[:, :h, :w]

        pooled = np.zeros((4, h//2, w//2))
        self.pool_indices = np.zeros((4, h//2, w//2, 2), dtype=int)

        for i in range(4):
            for r in range(0, h, 2):
                for c in range(0, w, 2):
                    # Safe slicing
                    region = self.feature_maps[i, r:r+2, c:c+2]
                    max_val = np.max(region)
                    pooled[i, r//2, c//2] = max_val
                    max_pos = np.unravel_index(np.argmax(region), region.shape)
                    self.pool_indices[i, r//2, c//2] = [r + max_pos[0], c + max_pos[1]]

        self.pooled = pooled
        flattened = self.pooled.flatten()

        # Dense Layer
        self.dense_input = flattened
        self.logits = np.dot(flattened, self.W_dense) + self.b_dense
        exp_logits = np.exp(self.logits - np.max(self.logits))
        self.output = exp_logits / np.sum(exp_logits)

        return self.output


    def backward(self, d_loss):
        # d_loss shape: (num_classes,)

        # --- 1. Backprop through Dense Layer ---
        # d_W_dense shape: (input_features, num_classes)
        d_W_dense = np.dot(self.dense_input.reshape(-1, 1), d_loss.reshape(1, -1))
        d_b_dense = d_loss

        # Gradient to pass back to the pooled layer
        # d_flat shape: (input_features,) e.g., (1024,)
        d_flat = np.dot(self.W_dense, d_loss)

        # Update Dense Weights
        self.W_dense -= self.lr * d_W_dense
        self.b_dense -= self.lr * d_b_dense

        # --- 2. Reshape gradient to match pooled output (4, 16, 16) ---
        d_pooled = d_flat.reshape(self.pooled.shape)

        # --- 3. Backprop through MaxPool ---
        d_feature_maps = np.zeros_like(self.feature_maps)
        h_out, w_out = d_pooled.shape[1], d_pooled.shape[2]

        for i in range(4): # For each filter
            for r in range(h_out):
                for c in range(w_out):
                    # Retrieve the original coordinates of the max value
                    orig_r, orig_c = self.pool_indices[i, r, c]
                    # Route the gradient only to the max position
                    d_feature_maps[i, orig_r, orig_c] += d_pooled[i, r, c]

        # --- 4. Backprop through ReLU ---
        # d_conv_outputs = d_feature_maps * (1 if conv_output > 0 else 0)
        # self.conv_outputs stores the pre-ReLU values from forward pass
        d_conv_outputs = d_feature_maps * (np.array(self.conv_outputs) > 0)

        # --- 5. Backprop through Conv2D ---
        # Initialize gradients
        d_filters = np.zeros_like(self.filters)
        d_input = np.zeros_like(self.input)

        for f_idx in range(4):
            h_out, w_out = d_conv_outputs[f_idx].shape

            # A. Calculate Gradient for Filters (dW)
            # Iterate over every output pixel to accumulate the gradient
            for i in range(h_out):
                for j in range(w_out):
                    # Extract the 3x3 region from the original input that created this output
                    # With padding=1, output(i,j) corresponds to input centered at (i,j)
                    region = np.zeros((3, 3))
                    for kr in range(3):
                        for kc in range(3):
                            ir, ic = i - 1 + kr, j - 1 + kc
                            if 0 <= ir < self.input.shape[0] and 0 <= ic < self.input.shape[1]:
                                region[kr, kc] = self.input[ir, ic]

                    # Accumulate: dW += input_region * output_gradient
                    d_filters[f_idx] += region * d_conv_outputs[f_idx, i, j]

            # B. Calculate Gradient for Input (d_input) to pass to previous layer
            # Perform full convolution with the 180-degree rotated filter
            rotated_filter = self.filters[f_idx][::-1, ::-1]

            # Note: Ensure your convolve2d supports mode='full'
            grad_slice = convolve2d(d_conv_outputs[f_idx], rotated_filter, mode='full')

            # Crop the result to match the original input size
            # 'full' convolution of (H, W) and (3, 3) results in (H+2, W+2)
            # We need to remove the 1-pixel border added by the convolution
            if grad_slice.shape[0] > self.input.shape[0]:
                pad = 1
                grad_slice = grad_slice[pad:-pad, pad:-pad]

            d_input += grad_slice

        # --- 6. Update Conv Weights and Biases ---
        self.filters -= self.lr * d_filters
        self.bias -= self.lr * np.sum(d_conv_outputs, axis=(1, 2))

        return d_input


    def train_step(self, x, y_true):
        # Forward
        y_pred_flat = self.forward(x)

        # Simple MSE Loss derivative for demonstration (d_loss = y_pred - y_true)
        # Note: For classification, you'd typically use CrossEntropy + Softmax
        # Here we assume y_true is flattened target vector same shape as y_pred_flat
        loss = 0.5 * np.sum((y_pred_flat - y_true)**2)
        d_loss = (y_pred_flat - y_true)

        # Backward
        self.backward(d_loss)
        return loss

# --- Example Usage ---
# Create dummy data (single 32x32 image, target is arbitrary vector)

def one_hot_encode(labels, num_classes):
    """Converts integer labels to one-hot vectors."""
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

def reshape_for_cnn(X_flat, img_size=(32, 32)):
    """Reshapes flattened data back to (samples, height, width) for CNN input."""
    return X_flat.reshape(-1, img_size[0], img_size[1])

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load Data
    # Ensure 'dataset' folder exists with subfolders 'hand' and 'no_hand'
    if not os.path.exists("dataset"):
        print("Error: 'dataset' folder not found. Run the capture script first.")
        exit()

    X_flat, y_labels, num_classes = load_dataset_from_folder("dataset", img_size=(32, 32))

    if len(X_flat) == 0:
        print("Error: No images loaded. Check folder contents.")
        exit()

    # 2. Preprocess
    # Reshape from (N, 1024) to (N, 32, 32) for CNN
    X_train = reshape_for_cnn(X_flat, (32, 32))

    # One-hot encode labels
    y_train_one_hot = one_hot_encode(y_labels, num_classes)

    print(f"Dataset Loaded: {X_train.shape[0]} images, {num_classes} classes.")
    print(f"Input Shape: {X_train.shape} | Output Shape: {y_train_one_hot.shape}")

    # 3. Initialize Model
    # Note: Your SimpleCNN class needs to match the input/output dimensions
    # If your SimpleCNN is hardcoded for specific filter counts, ensure it matches here.
    # For this example, we assume SimpleCNN handles the dimensions dynamically or matches 32x32 input.
    input_features = 1024
    num_classes = 2  # "hand" and "no_hand"

    # Initialize with arguments
    model = SimpleCNN(input_features=input_features, num_classes=num_classes)

    # 4. Training Loop
    epochs = 5
    print(f"\nStarting Training for {epochs} epochs...")

    for epoch in range(epochs):
        total_loss = 0
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train_one_hot = y_train_one_hot[indices]

        for i in range(len(X_train)):
            # Forward
            prediction = model.forward(X_train[i])

            # Loss (MSE)
            loss = 0.5 * np.sum((prediction - y_train_one_hot[i])**2)
            total_loss += loss

            # Backward
            d_loss = (prediction - y_train_one_hot[i])
            model.backward(d_loss)

        avg_loss = total_loss / len(X_train)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    print("\nTraining Complete! Model ready for real-time testing.")

    # 5. Real-Time Testing (Optional)
    # Uncomment to test live after training
    """
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        input_img = cv2.resize(gray, (32, 32)) / 255.0

        output = model.forward(input_img)
        pred_class = np.argmax(output)
        conf = np.max(output)

        classes = sorted([d for d in os.listdir("dataset") if os.path.isdir(os.path.join("dataset", d))])
        label_name = classes[pred_class] if pred_class < len(classes) else "Unknown"

        cv2.putText(frame, f"{label_name}: {conf:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Live Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()
    """
def run_real_time_test(model, dataset_folder="dataset", img_size=(32, 32)):
    print("\n=== STARTING REAL-TIME TEST ===")
    print("Point camera at objects. Press 'q' to quit.")

    # Get class names (sorted alphabetically to match training)
    classes = sorted([d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))])
    print(f"Detecting classes: {classes}")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Preprocess Frame (Must match training exactly)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, img_size)
        input_data = resized / 255.0  # Normalize

        # 2. Predict
        # model.forward expects a 2D array (H, W)
        output_probs = model.forward(input_data)

        # 3. Interpret Results
        predicted_idx = np.argmax(output_probs)
        confidence = np.max(output_probs)

        if predicted_idx < len(classes):
            label = classes[predicted_idx]
        else:
            label = "Unknown"

        # 4. Display on Screen
        text = f"{label}: {confidence:.2f}"
        color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255) # Green if confident, Red if not

        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("DIY CNN Live Feed", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Execution ---
# After your training loop finishes, call this function:
run_real_time_test(model)




img = np.random.rand(32, 32)
target = np.random.rand(4 * (16*16)) # Flattened pooled output size

cnn = SimpleCNN()
loss = cnn.train_step(img, target)
print(f"Initial Loss: {loss:.4f}")

