
import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt

# ✅ Lambda katmanları için tanımlar
def l2_normalize(t, **kwargs):
    return tf.math.l2_normalize(t, axis=1)

def euclidean_distance(tensors, **kwargs):
    return tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True)

# ✅ Modeli yükle (.keras formatı ile)
model = tf.keras.models.load_model(
    "models/siamese_graphology_model.keras",
    custom_objects={
        "l2_normalize": l2_normalize,
        "euclidean_distance": euclidean_distance
    },
    compile=False
)

# ✅ Embedding çıkaran alt modeli yakala
embedder = model.get_layer("embedding_model")

# ✅ Görselden embedding çıkar
def get_embedding(img_path, size=(224, 224)):
    print(f"👉 Embed çıkarılıyor: {img_path}")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Görsel bulunamadı: {img_path}")
    img = cv2.imread(img_path)
    img = cv2.resize(img, size)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    print("🧠 embedder.predict_on_batch() çağrılıyor...")
    embedding = embedder.predict_on_batch(img)
    print("✅ Embedding üretildi.")
    return embedding[0]

# ✅ Cosine similarity hesapla
def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

# ✅ Referans embedding'leri oluştur
def build_reference_embeddings(reference_dir="data/processed", samples_per_class=3):
    reference = {}
    for class_name in sorted(os.listdir(reference_dir)):
        class_path = os.path.join(reference_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        embeddings = []
        for fname in sorted(os.listdir(class_path))[:samples_per_class]:
            path = os.path.join(class_path, fname)
            emb = get_embedding(path)
            embeddings.append(emb)
        if embeddings:
            reference[class_name] = embeddings
    return reference

# 🔍 Top-k tahmin
def predict_top_k(test_image_path, reference_embeddings, top_k=3):
    test_emb = get_embedding(test_image_path)
    scores = []
    for class_name, embs in reference_embeddings.items():
        similarities = [cosine_similarity(test_emb, ref_emb) for ref_emb in embs]
        avg_sim = np.mean(similarities)
        scores.append((class_name, avg_sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

# ✅ Ana çalışma
if __name__ == "__main__":
    reference_embeddings = build_reference_embeddings(samples_per_class=5)
    test_image = "test_samples/argumentative.jpeg"


    # 🔍 Gözlemsel analiz için en benzer 3 sınıf
    top3 = predict_top_k(test_image, reference_embeddings, top_k=5)
    print("\n🔎 Top-3 Benzer Sınıf:")
    for name, sim in top3:
        print(f"  - {name}: {sim:.2f}")

    # Görsel önizleme
    img = cv2.imread(test_image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title("Test Görseli")
    plt.axis("off")
    plt.show()