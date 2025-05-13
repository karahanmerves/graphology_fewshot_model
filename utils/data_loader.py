import tensorflow as tf
import os
import random

AUTOTUNE = tf.data.AUTOTUNE

# 🔹 Augmentasyon sadece pozitif çiftlerde kullanılacak
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)
], name="augmentation")

# 🔹 Tek görseli oku ve işle
def preprocess_image(filename, augment=False):
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [224, 224])  # Önceden resize yapılmamışsa
    image = tf.cast(image, tf.float32) / 255.0
    if augment:
        image = augmentation(image)
    return image

# 🔹 Tüm görsel yollarını ve etiketleri yükle
def load_image_paths(data_dir):
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    label_to_index = {name: i for i, name in enumerate(class_names)}

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_path):
            if fname.endswith(".png"):
                image_paths.append(os.path.join(class_path, fname))
                labels.append(label_to_index[class_name])
    
    return image_paths, labels, label_to_index

# 🔹 Pozitif ve negatif çiftleri oluştur, augmentasyon uygula
def create_pair_dataset(image_paths, labels, pairs_per_class=10, buffer_size=1000):
    pairs = []
    labels_out = []

    label_to_paths = {}
    for path, label in zip(image_paths, labels):
        label_to_paths.setdefault(label, []).append(path)

    class_labels = list(label_to_paths.keys())

    for cls in class_labels:
        images = label_to_paths[cls]
        if len(images) < 2:
            continue

        # ✅ Pozitif çiftler
        for _ in range(pairs_per_class):
            a, b = random.sample(images, 2)
            pairs.append((a, b, 1))

        # ✅ Negatif çiftler
        other_classes = [l for l in class_labels if l != cls]
        for _ in range(pairs_per_class):
            a = random.choice(images)
            negative_cls = random.choice(other_classes)
            b = random.choice(label_to_paths[negative_cls])
            pairs.append((a, b, 0))

    random.shuffle(pairs)

    def preprocess_pair(path1, path2, label):
        is_positive = tf.equal(label, 1)
        img1 = preprocess_image(path1, augment=is_positive)
        img2 = preprocess_image(path2, augment=is_positive)
        return (img1, img2), label

    path1_ds = tf.data.Dataset.from_tensor_slices([p[0] for p in pairs])
    path2_ds = tf.data.Dataset.from_tensor_slices([p[1] for p in pairs])
    label_ds = tf.data.Dataset.from_tensor_slices([p[2] for p in pairs])

    ds = tf.data.Dataset.zip((path1_ds, path2_ds, label_ds))
    ds = ds.shuffle(buffer_size).map(preprocess_pair, num_parallel_calls=AUTOTUNE)
    return ds

