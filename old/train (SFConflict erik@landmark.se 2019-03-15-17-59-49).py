import data
import plot
import image_processing
import pathlib
import tensorflow as tf

tf.enable_eager_execution()

root = pathlib.Path("./flower_photos")
image_count = len(image_processing.get_all_image_paths(root))
dataset = data.get_data_set_from_path(root)

BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.
ds = dataset.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# `prefetch` lets the dataset fetch batches, in the background while the model is training.
ds = ds.prefetch(buffer_size=AUTOTUNE)

mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable=False

image_batch, label_batch = next(iter(ds))
feature_map_batch = mobile_net(image_batch)

model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(image_processing.get_labels_name(root)))])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])
model.summary()

steps_per_epoch = int(tf.ceil(image_count / BATCH_SIZE).numpy())

model.fit(ds, epochs=10, steps_per_epoch=steps_per_epoch)