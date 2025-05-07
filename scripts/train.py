import tensorflow as tf
from utils.data_loader import load_image_paths, create_pair_dataset
from utils.image_utils import cache_resized_images
from tensorflow.keras import layers, models, losses, optimizers
import os

# 📌 1. İşlenmiş veri yoksa önce cachele
if not os.path.exists("data/processed"):
    cache_resized_images("data/raw", "data/processed")

# 📌 2. Görsellerin yollarını ve etiketlerini al
image_paths, labels, label_to_index = load_image_paths("data/processed")

# 📌 3. Eğitim verisi çiftlerini oluştur
dataset = create_pair_dataset(image_paths, labels).batch(32)

# 📌 4. Embedding Model (MobileNetV2 tabanlı)
def build_embedding_model(input_shape=(224, 224, 3)):
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
    return tf.keras.Model(inputs, x)

# 📌 5. Siamese Model
def build_siamese_model(input_shape=(224, 224, 3)):
    embedder = build_embedding_model(input_shape)
    input_a = tf.keras.Input(shape=input_shape)
    input_b = tf.keras.Input(shape=input_shape)
    vec_a = embedder(input_a)
    vec_b = embedder(input_b)
    distance = layers.Lambda(
        lambda tensors: tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True)
    )([vec_a, vec_b])
    return tf.keras.Model(inputs=[input_a, input_b], outputs=distance)

# 📌 6. Contrastive Loss
def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(
        y_true * tf.square(y_pred) +
        (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
    )

# 📌 7. Modeli derle ve eğit
model = build_siamese_model()
model.compile(optimizer=optimizers.Adam(1e-4), loss=contrastive_loss)

model.fit(dataset, epochs=10)

# 📌 8. Kaydet
model.save("models/siamese_graphology_model.h5")
print("✅ Model eğitildi ve kaydedildi.")
