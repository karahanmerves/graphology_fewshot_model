import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.applications.efficientnet import preprocess_input

# Lambda fonksiyonları
def l2_normalize(x):
    return tf.math.l2_normalize(x, axis=1)

def euclidean_distance(vectors):
    x, y = vectors
    return tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)

# EfficientNet tabanlı modeli yükle
model = tf.keras.models.load_model(
    "models/siamese_graphology_model.keras",
    custom_objects={
        "l2_normalize": l2_normalize,
        "euclidean_distance": euclidean_distance
    },
    compile=False
)
embedder = model.get_layer("embedding_model")

# Embedding çıkar
def get_embedding(img_path, size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return embedder.predict(img, verbose=0)[0]

# Embedding'leri ve etiketleri topla
base_dir = "data/processed"
embeddings = []
labels = []

for class_name in sorted(os.listdir(base_dir)):
    class_path = os.path.join(base_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    for fname in sorted(os.listdir(class_path))[:3]:  # samples_per_class = 3
        fpath = os.path.join(class_path, fname)
        try:
            emb = get_embedding(fpath)
            embeddings.append(emb)
            labels.append(class_name)
        except Exception as e:
            print(f"Hata: {fpath} -> {e}")

# PCA ile 2D'ye indir
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

# Görselleştir
plt.figure(figsize=(12, 8))
unique_labels = sorted(set(labels))
colors = plt.cm.get_cmap("tab20", len(unique_labels))

for i, label in enumerate(unique_labels):
    idxs = [j for j, l in enumerate(labels) if l == label]
    plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=label, alpha=0.7, s=80)

plt.title("📌 PCA ile Graphology Embedding Haritası (EfficientNet)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
