import pathlib
import tensorflow as tf


def get_labels_name(path):
    return sorted(item.name for item in path.glob('*/') if item.is_dir())


def get_all_image_paths(path):
    all_image_paths = list(path.glob('*/*'))
    all_image_paths = [str(p) for p in all_image_paths]
    return all_image_paths


def get_all_images_labels(path):
    label_names = get_labels_name(path)
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in get_all_image_paths(path)]

    return all_image_labels


def load_and_preprocess_image(path):
    return tf.image.resize_images(tf.image.decode_jpeg(tf.io.read_file(path), channels=3), [192, 192]) / 255.0
