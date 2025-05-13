import os
import numpy as np
import tensorflow as tf
import cv2

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
    img = cv2.imread(img_path)
    img = cv2.resize(img, size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    embedding = embedder.predict(img)
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
        embeddings = []
        for fname in os.listdir(class_path)[:samples_per_class]:
            path = os.path.join(class_path, fname)
            emb = get_embedding(path)
            embeddings.append(emb)
        reference[class_name] = embeddings
    return reference

# ✅ Tahmin fonksiyonu
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

# ✅ Örnek kullanım
if __name__ == "__main__":
    reference_embeddings = build_reference_embeddings()
    test_image = "test_samples/test_sample_resized.png"
    label, score = predict_label(test_image, reference_embeddings)
    print(f"📍 Tahmin: {label} ({score:.2f})")
