import tensorflow as tf
import plot
import timeit

tf.enable_eager_execution()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(graph=tf.get_default_graph(), config=config)
sess.as_default()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

LABELS = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def resize_fn(img):
    tens = tf.tile(tf.expand_dims(tf.convert_to_tensor(img), 2), [1, 1, 3])
    return tf.image.resize_images([tens], tf.convert_to_tensor([96, 96]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[0]


x_test_ds = tf.cast(tf.convert_to_tensor(tf.map_fn(resize_fn, x_test[:100])), 'float32') / 255.0
# y_test_ds = tf.keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')
test_ds = tf.data.Dataset.from_tensor_slices(x_test_ds)
test_ds = test_ds.batch(64)
test_ds = test_ds.cache(filename='./cache/cache-inf.tf-data')

model = tf.keras.models.load_model('models/model.h5')
model.summary()


tf.global_variables_initializer()


def pre():

    rights = 0
    
    for image_nr in range(10000):
        image = x_test_ds[image_nr]
        label = model.predict(tf.expand_dims(image, 0), steps=1, verbose=1)
        argmax = tf.math.argmax(label[0]).numpy()
        
        #print(f"{LABELS[argmax]} = {LABELS[y_test[image_nr]]}")
        
        if y_test[image_nr] == argmax:
            rights += 1

        # title = f"Predict: {LABELS[argmax]} | Right: {LABELS[y_test[image_nr]]}"
        # plot.plot_image_from_tensor(image, title=title)
    
    print(f"Got {rights} of 10000 rights! ({rights / 10000 * 100} %)")


time = timeit.timeit(pre, number=1)
print(time)



