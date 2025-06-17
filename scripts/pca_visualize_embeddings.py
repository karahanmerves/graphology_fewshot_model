import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 🔹 Lambda katmanlarını tekrar tanımla
def l2_normalize(t, **kwargs):
    return tf.math.l2_normalize(t, axis=1)

def euclidean_distance(tensors, **kwargs):
    return tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True)

# 🔹 Modeli yükle
model = tf.keras.models.load_model(
    "models/siamese_graphology_model.keras",
    custom_objects={
        "l2_normalize": l2_normalize,
        "euclidean_distance": euclidean_distance
    },
    compile=False
)
embedder = model.get_layer("embedding_model")

# 🔹 Görselden embedding çıkar
def get_embedding(img_path, size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    embedding = embedder.predict(img, verbose=0)
    return embedding[0]

# 🔹 Embedding'leri oluştur
base_dir = "data/processed"
all_embeddings = []
all_labels = []

for class_name in sorted(os.listdir(base_dir)):
    class_path = os.path.join(base_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    for fname in sorted(os.listdir(class_path))[:3]:
        path = os.path.join(class_path, fname)
        try:
            emb = get_embedding(path)
            all_embeddings.append(emb)
            all_labels.append(class_name)
        except Exception as e:
            print(f"Hata: {path} | {e}")

# 🔹 PCA görselleştirme
pca = PCA(n_components=2)
reduced = pca.fit_transform(all_embeddings)

plt.figure(figsize=(12, 8))
unique_labels = sorted(set(all_labels))
colors = plt.cm.get_cmap('tab20', len(unique_labels))

for idx, label in enumerate(unique_labels):
    indices = [i for i, l in enumerate(all_labels) if l == label]
    plt.scatter(reduced[indices, 0], reduced[indices, 1], label=label, alpha=0.7, s=70)

plt.title("📌 PCA ile Graphology Embedding Haritası")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
