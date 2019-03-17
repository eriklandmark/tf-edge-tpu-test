import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model_file('models/model.h5')
tfmodel = converter.convert()
open("models/model.tflite", "wb").write(tfmodel)
