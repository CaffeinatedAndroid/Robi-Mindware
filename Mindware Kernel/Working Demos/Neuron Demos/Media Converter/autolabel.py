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
X_train, y_train, num_classes = load_dataset_from_folder()

# # 3. One-hot encode labels for your CNN
# def one_hot_encode(labels, num_classes):
#      one_hot = np.zeros((labels.size, num_classes))
#      one_hot[np.arange(labels.size), labels] = 1
#      return one_hot
# y_train_one_hot = one_hot_encode(y_train, num_classes)
