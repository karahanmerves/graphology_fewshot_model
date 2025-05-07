import tensorflow as tf
import os
import random

AUTOTUNE = tf.data.AUTOTUNE

def preprocess_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # Resize yok çünkü önceden yapıldı
    return image

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

def create_pair_dataset(image_paths, labels, buffer_size=1000):
    pairs = []
    labels_out = []
    
    label_to_paths = {}
    for path, label in zip(image_paths, labels):
        label_to_paths.setdefault(label, []).append(path)

    for anchor_path, anchor_label in zip(image_paths, labels):
        # Pozitif çift
        positive_path = random.choice([p for p in label_to_paths[anchor_label] if p != anchor_path])
        pairs.append((anchor_path, positive_path))
        labels_out.append(1)

        # Negatif çift
        negative_label = random.choice([l for l in label_to_paths.keys() if l != anchor_label])
        negative_path = random.choice(label_to_paths[negative_label])
        pairs.append((anchor_path, negative_path))
        labels_out.append(0)

    def preprocess_pair(path1, path2, label):
        img1 = preprocess_image(path1)
        img2 = preprocess_image(path2)
        return (img1, img2), label

    path_ds_1 = tf.data.Dataset.from_tensor_slices([p[0] for p in pairs])
    path_ds_2 = tf.data.Dataset.from_tensor_slices([p[1] for p in pairs])
    label_ds = tf.data.Dataset.from_tensor_slices(labels_out)

    ds = tf.data.Dataset.zip((path_ds_1, path_ds_2, label_ds))
    ds = ds.shuffle(buffer_size).map(preprocess_pair, num_parallel_calls=AUTOTUNE)
    return ds
