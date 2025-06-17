import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input

# Lambda fonksiyonları
def l2_normalize(t, **kwargs):
    return tf.math.l2_normalize(t, axis=1)

def euclidean_distance(tensors, **kwargs):
    return tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True)

# Modeli yükle
model = tf.keras.models.load_model(
    "models/siamese_graphology_model.keras",
    custom_objects={
        "l2_normalize": l2_normalize,
        "euclidean_distance": euclidean_distance
    },
    compile=False
)
embedder = model.get_layer("embedding_model")

# Görselden embedding çıkar
def get_embedding(img_path, size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, size)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    emb = embedder.predict(img, verbose=0)
    return emb[0]

# Cosine similarity
def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

# Referans embedding’leri oluştur
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

# Top-k tahmin
def predict_top_k(test_emb, reference_embeddings, top_k=5):
    scores = []
    for class_name, embs in reference_embeddings.items():
        sims = [cosine_similarity(test_emb, ref_emb) for ref_emb in embs]
        avg_sim = np.mean(sims)
        scores.append((class_name, avg_sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

# Dosya adından sınıf ismini çıkar
def extract_label(filename):
    name = os.path.splitext(filename)[0].lower()
    return name if "_" not in name else "_".join(name.split("_")[:3])

# Değerlendirme
def evaluate_all(test_dir="test_samples", reference_dir="data/processed", top_k=1):
    reference_embeddings = build_reference_embeddings(reference_dir)
    results = []

    for fname in sorted(os.listdir(test_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        fpath = os.path.join(test_dir, fname)
        expected = extract_label(fname)
        test_emb = get_embedding(fpath)
        top_preds = predict_top_k(test_emb, reference_embeddings, top_k)

        predicted, score = top_preds[0]
        is_correct = expected == predicted
        in_top_k = any(expected == label for label, _ in top_preds)

        results.append({
            "File": fname,
            "Expected": expected,
            "Top-1": predicted,
            "Score": round(score, 4),
            "Top-K Match": "✔️" if in_top_k else "❌"
        })

    # Terminal çıktısı
    print(f"\n{'File':<30} {'Expected':<25} {'Top-1':<25} {'Score':<8} {'Top-K Match'}")
    print("-" * 95)
    for r in results:
        print(f"{r['File']:<30} {r['Expected']:<25} {r['Top-1']:<25} {r['Score']:<8} {r['Top-K Match']}")

if __name__ == "__main__":
    evaluate_all()
