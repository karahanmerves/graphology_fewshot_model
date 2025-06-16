import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from PIL import Image

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
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Görsel bulunamadı: {img_path}")
    img = cv2.imread(img_path)
    img = cv2.resize(img, size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    embedding = embedder.predict(img, verbose=0)
    return embedding[0]

# ✅ Cosine similarity hesapla
def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

# ✅ Referans embedding'leri oluştur
def build_reference_embeddings(reference_dir="data/processed", samples_per_class=3):
    reference = {}
    if not os.path.exists(reference_dir):
        raise FileNotFoundError(f"Referans klasörü bulunamadı: {reference_dir}")
    
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

print("\n🧪 ARGUMENTATIVE sınıfı örnek embedding analizi:")
argu_dir = "data/processed/argumentative"
for fname in sorted(os.listdir(argu_dir))[:3]:
    fpath = os.path.join(argu_dir, fname)
    emb = get_embedding(fpath)
    print(f"→ {fname} | mean: {np.mean(emb):.6f}, norm: {np.linalg.norm(emb):.6f}")

# ✅ Tek sonuç döndüren karar fonksiyonu
def predict_label(test_image_path, reference_embeddings, threshold=0.75):
    test_emb = get_embedding(test_image_path)
    best_class = None
    best_score = -1

    for class_name, embs in reference_embeddings.items():
        similarities = [cosine_similarity(test_emb, ref_emb) for ref_emb in embs]
        avg_sim = np.mean(similarities)
        if avg_sim > best_score:
            best_score = avg_sim
            best_class = class_name

    if best_score >= threshold:
        return best_class, best_score
    else:
        return "unknown", best_score

# 🔍 Gözlemsel analiz için top-k sıralı sınıflar
def predict_top_k(test_image_path, reference_embeddings, top_k=3):
    test_emb = get_embedding(test_image_path)
    scores = []

    for class_name, embs in reference_embeddings.items():
        similarities = [cosine_similarity(test_emb, ref_emb) for ref_emb in embs]
        avg_sim = np.mean(similarities)
        scores.append((class_name, avg_sim))

    # En yüksek benzerlik sırasına göre sırala
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

# 🔹 Test görselini çiz
def show_test_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Test Image")
    plt.show()


# ✅ Örnek kullanım
if __name__ == "__main__":
    reference_embeddings = build_reference_embeddings(samples_per_class=5)
    test_image = "test_samples/test_sample_4.png"


    # 🧠 Üretim amaçlı tek sınıf tahmini
    label, score = predict_label(test_image, reference_embeddings, threshold=0.85)
    print(f"📍 Tahmin: {label} (Benzerlik: {score:.2f})")

    # 🔍 Gözlemsel analiz için en benzer 3 sınıf
    top3 = predict_top_k(test_image, reference_embeddings, top_k=10)
    print("\n🔎 Top-10 Benzer Sınıf:")
    for name, sim in top3:
        print(f"  - {name}: {sim:.2f}")

    print("\n🖼 Test görseli:")
    show_test_image(test_image)


