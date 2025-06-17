
import tensorflow as tf
from tensorflow.keras import layers

# ✅ Lambda katmanları
def l2_normalize(x):
    return tf.math.l2_normalize(x, axis=1)

def euclidean_distance(vectors):
    x, y = vectors
    return tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)

# ✅ Embedding Model (EfficientNet tabanlı)
def build_embedding_model(input_shape=(224, 224, 3)):
    base = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape, include_top=False, weights="imagenet", pooling="avg"
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Lambda(l2_normalize, name="l2_normalize")(x)
    return tf.keras.Model(inputs, x, name="embedding_model")

# ✅ Siamese (Protonet tarzı) Model
def build_siamese_model(input_shape=(224, 224, 3)):
    embedder = build_embedding_model(input_shape)
    input_a = tf.keras.Input(shape=input_shape)
    input_b = tf.keras.Input(shape=input_shape)
    vec_a = embedder(input_a)
    vec_b = embedder(input_b)
    distance = layers.Lambda(euclidean_distance, name="euclidean_distance")([vec_a, vec_b])
    return tf.keras.Model(inputs=[input_a, input_b], outputs=distance)

# Bu sadece model tanımıdır. Eğitim, optimizer, loss ve callback'leri aşağıda ayrıca bağlaman gerekir.
