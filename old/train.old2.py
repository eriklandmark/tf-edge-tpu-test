import data
import plot
import image_processing
import pathlib
import tensorflow as tf
import math

tf.enable_eager_execution()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train_ds = x_train.cast('float32') / 255
x_test_ds = x_test.cast('float32') / 255
y_train_ds = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
y_test_ds = tf.keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')

# root = pathlib.Path("./flower_photos")
# image_count = len(image_processing.get_all_image_paths(root))
# dataset = data.get_data_set_from_path(root)

BATCH_SIZE = 200
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMAGE_COUNT = 60000


def shuffle_dataset(ds):
    return ds.shuffle(buffer_size=IMAGE_COUNT).repeat().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)


x_train_ds = shuffle_dataset(x_train_ds)

mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3),
                                               include_top=False)
mobile_net.trainable = True

image_batch, label_batch = next(iter(x_train))
feature_map_batch = mobile_net(image_batch)

model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(image_processing.get_labels_name(root)))])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])
model.summary()

# with tf.Session() as sess:
tf.global_variables_initializer()
steps_per_epoch = int(math.ceil(IMAGE_COUNT / BATCH_SIZE))
model.fit(x_train_ds, epochs=20, steps_per_epoch=steps_per_epoch)
model.save("saved_model.h5")

# s_model = tf.keras.models.load_model("saved_model.h5")
# s_model.evaluate(ds, steps=10)

