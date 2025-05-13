import tensorflow as tf
from utils.data_loader import load_image_paths, create_pair_dataset
from utils.image_utils import cache_resized_images
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# 🔹 Lambda fonksiyonları
def l2_normalize(t):
    return tf.math.l2_normalize(t, axis=1)

def euclidean_distance(tensors):
    return tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True)

# 🔹 Minimal Augmentasyon pipeline (yalnızca flip ve zoom)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(0.1)
], name="simple_augmentation")

# 🔸 1. İşlenmiş veri yoksa cachele
if not os.path.exists("data/processed"):
    cache_resized_images("data/raw", "data/processed")

# 🔸 2. Veri yollarını ve etiketleri al
image_paths, labels, label_to_index = load_image_paths("data/processed")

# 🔸 3. Dataset oluştur (daha az çift)
dataset = create_pair_dataset(image_paths, labels, pairs_per_class=20)
dataset = dataset.shuffle(1000)

# 🔸 4. Eğitim / validation bölmesi
total = tf.data.experimental.cardinality(dataset).numpy()
val_size = int(0.2 * total)
train_dataset = dataset.skip(val_size).batch(32)
val_dataset = dataset.take(val_size).batch(32)

# 🔸 5. Embedding modeli (MobilNetV2 tamamen freeze)
def build_embedding_model(input_shape=(224, 224, 3)):
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet", pooling="avg"
    )
    base.trainable = False  # Tamamen dondur

    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Lambda(l2_normalize, name="l2_normalize")(x)
    return tf.keras.Model(inputs, x, name="embedding_model")

# 🔸 6. Siamese model
def build_siamese_model(input_shape=(224, 224, 3)):
    embedder = build_embedding_model(input_shape)
    input_a = tf.keras.Input(shape=input_shape)
    input_b = tf.keras.Input(shape=input_shape)
    vec_a = embedder(input_a)
    vec_b = embedder(input_b)
    distance = layers.Lambda(euclidean_distance, name="euclidean_distance")([vec_a, vec_b])
    return tf.keras.Model(inputs=[input_a, input_b], outputs=distance)

# 🔸 7. Contrastive loss
def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(
        y_true * tf.square(y_pred) +
        (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
    )

# 🔸 8. Callback'ler (erken durdurma 2 epoch ile)
callbacks = [
    EarlyStopping(patience=2, monitor="val_loss", restore_best_weights=True),
    ModelCheckpoint("models/best_siamese.keras", save_best_only=True, monitor="val_loss")
]

# 🔸 9. Model oluştur ve eğit
model = build_siamese_model()
model.compile(
    optimizer=optimizers.Adam(1e-4),
    loss=contrastive_loss
)
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=callbacks
)

# 🔸 10. Model kaydet (.keras ve .tflite)
model.save("models/siamese_graphology_model.keras")
print("✅ Model .keras formatında kaydedildi.")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("models/siamese_graphology_model.tflite", "wb") as f:
    f.write(tflite_model)
print("📦 Model .tflite formatında da kaydedildi.")

