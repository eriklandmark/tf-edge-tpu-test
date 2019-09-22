import tensorflow as tf

# tf.enable_eager_execution()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()


def resize_fn(img):
    tens = tf.tile(tf.expand_dims(tf.convert_to_tensor(img), 2), [1, 1, 3])
    return tf.image.resize_images([tens], tf.convert_to_tensor([96, 96]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[0]


x_test_reshape = tf.map_fn(resize_fn, x_test)
x_test_ds = tf.cast(tf.convert_to_tensor(x_test_reshape), 'float32') / 255.0
y_test_ds = tf.keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')
test_ds = tf.data.Dataset.from_tensor_slices((x_test_ds, y_test_ds))
test_ds = test_ds.batch(10000)
test_ds = test_ds.cache(filename='./cache/cache-eval.tf-data')

model = tf.keras.models.load_model('models/model.h5')
model.summary()

loss, acc = model.evaluate(test_ds, steps=10)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))