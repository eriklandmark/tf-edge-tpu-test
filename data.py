import tensorflow as tf
import pathlib
import image_processing


def get_data_set_from_path(ds_path):
    ds_path = pathlib.Path(ds_path)
    all_image_paths = image_processing.get_all_image_paths(ds_path)
    all_image_labels = image_processing.get_all_images_labels(ds_path)
    ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
    image_label_ds = ds.map(lambda path, label: (image_processing.load_and_preprocess_image(path), label))
    image_label_ds = image_label_ds.map(lambda image, label: (2 * image - 1, label))
    return image_label_ds
