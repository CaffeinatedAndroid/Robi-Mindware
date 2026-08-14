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
create_and_save_dataset()



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

