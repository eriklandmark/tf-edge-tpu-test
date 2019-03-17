import tensorflow as tf
import math


def resize_fn(img):
    tens = tf.tile(tf.expand_dims(tf.convert_to_tensor(img), 2), [1, 1, 3])
    return tf.image.resize_images([tens], tf.convert_to_tensor([96, 96]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[0]


LABELS = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
IMAGE_COUNT = 60000
BATCH_SIZE = 64
AUTOTUNE = tf.data.experimental.AUTOTUNE
STEPS_PER_EPOCH = int(math.ceil(IMAGE_COUNT / BATCH_SIZE))
NUM_CLASSES = 10

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# tf.enable_eager_execution()


def data_generator(sess, x, y):
    try:
        for i in range(0, IMAGE_COUNT, BATCH_SIZE):
            x_data = x[i:i + BATCH_SIZE]
            x_data = tf.map_fn(resize_fn, x_data)
            x_data = tf.cast(tf.convert_to_tensor(x_data), 'float32') / 255.0
            y_data = tf.keras.utils.to_categorical(y[i:i + BATCH_SIZE], num_classes=NUM_CLASSES, dtype='float32')
            print(x_data.shape)
            yield (x_data, y_data)
    except Exception as err:
        print(err)


x_train_reshape = tf.map_fn(resize_fn, x_train)
x_train_ds = tf.cast(tf.convert_to_tensor(x_train_reshape), 'float32') / 255.0

y_train_ds = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
y_test_ds = tf.keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')

ds = tf.data.Dataset.from_tensor_slices((x_train_ds, y_train_ds))
ds = ds.cache(filename='./cache/cache.tf-data')
ds_batches = ds.shuffle(buffer_size=IMAGE_COUNT).repeat().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

mobile_net = tf.keras.applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False, pooling='avg', weights='imagenet')
mobile_net.trainable = True

model = tf.keras.Sequential(
    [
        # tf.keras.layers.Input(shape=(96, 96, 3)),
        mobile_net,
        tf.keras.layers.Dense(256, activation='relu'),
        # tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(10, activation='softmax')
    ]
)
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

model.summary()
tf.global_variables_initializer()
model.fit(ds_batches, epochs=5, steps_per_epoch=STEPS_PER_EPOCH, verbose=1)
model.save("model.h5")
