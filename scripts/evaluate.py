import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 🔹 Lambda katmanları
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
    img = load_img(img_path, target_size=size)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return embedder.predict(img, verbose=0)[0]

# 🔹 Cosine similarity
def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

# 🔹 Referans embedding'leri hazırla
def build_reference_embeddings(reference_dir="data/processed", samples_per_class=3):
    reference = {}
    for class_name in sorted(os.listdir(reference_dir)):
        class_path = os.path.join(reference_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        embeddings = []
        for fname in sorted(os.listdir(class_path))[:samples_per_class]:
            fpath = os.path.join(class_path, fname)
            emb = get_embedding(fpath)
            embeddings.append(emb)
        if embeddings:
            reference[class_name] = embeddings
    return reference

# 🔹 Dosya adından sınıfı çıkar (gelişmiş)
def extract_label(filename):
    name = os.path.splitext(filename)[0].lower()
    parts = name.split("_")
    if parts[0] in ["sensitive", "desire", "attentive"]:
        return "_".join(parts[:3])  # örn: sensitive_to_criticism
    return parts[0]  # örn: argumentative_2 → argumentative

# 🔹 Değerlendirme
def evaluate(test_dir="test_samples", reference_embeddings=None):
    results = []
    for fname in sorted(os.listdir(test_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        test_path = os.path.join(test_dir, fname)
        true_label = extract_label(fname)
        test_emb = get_embedding(test_path)

        best_score = -1
        best_class = None
        for class_name, embs in reference_embeddings.items():
            sims = [cosine_similarity(test_emb, ref_emb) for ref_emb in embs]
            avg_sim = np.mean(sims)
            if avg_sim > best_score:
                best_score = avg_sim
                best_class = class_name

        correct = best_class == true_label
        results.append((fname, true_label, best_class, round(best_score, 4), "✔️" if correct else "❌"))

    return results

# 🔹 Ana çalıştırma
if __name__ == "__main__":
    reference_embeddings = build_reference_embeddings(samples_per_class=3)
    results = evaluate(reference_embeddings=reference_embeddings)

    print(f"\n{'File':<30} {'Expected':<25} {'Predicted':<25} {'Score':<8} {'Result'}")
    print("-" * 95)
    for file, expected, predicted, score, check in results:
        print(f"{file:<30} {expected:<25} {predicted:<25} {score:<8} {check}")
