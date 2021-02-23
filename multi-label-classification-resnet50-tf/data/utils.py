import tensorflow
if tensorflow.version.VERSION[0] == '1':
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_eager_execution()

import math




# ------------------------------------------------------------------
def data_augment(image, size):
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_crop = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    # Flips
    if p_spatial >= .2:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

    # Rotates
    if p_rotate > .75:
        image = tf.image.rot90(image, k=3)  # rotate 270º
    elif p_rotate > .5:
        image = tf.image.rot90(image, k=2)  # rotate 180º
    elif p_rotate > .25:
        image = tf.image.rot90(image, k=1)  # rotate 90º

    # Crops
    if p_crop > .4:
        crop_size = tf.random.uniform([], int(size * .7), size, dtype=tf.int32)
        image = tf.image.random_crop(image, size=[crop_size, crop_size, 3])
    elif p_crop > .7:
        if p_crop > .9:
            image = tf.image.central_crop(image, central_fraction=.7)
        elif p_crop > .8:
            image = tf.image.central_crop(image, central_fraction=.8)
        else:
            image = tf.image.central_crop(image, central_fraction=.9)

    image = tf.image.resize(image, size=[size, size])

    # Pixel-level transforms
    if p_pixel >= .2:
        if p_pixel >= .8:
            image = tf.image.random_saturation(image, lower=0, upper=2)
        elif p_pixel >= .6:
            image = tf.image.random_contrast(image, lower=.8, upper=2)
        elif p_pixel >= .4:
            image = tf.image.random_brightness(image, max_delta=.2)
        else:
            image = tf.image.adjust_gamma(image, gamma=.6)

    return image


# ------------------------------------------------------------------
def _parse_function(filename, label, n_channels, size, is_training = True):
    """
    Returns resized and normalized image and its label
    """
    resized_image = _parse_image(filename, n_channels, size, is_training)
    return resized_image, label


def _parse_image(filename, n_channels, size, is_training = True):
    """Obtain the image from the filename (for both training and validation).
    The following operations are applied:
        - Decode the image from jpeg format
        - Convert to float and to range [0, 1]
    """
    image_string = tf.read_file(filename)

    image_decoded = tf.image.decode_jpeg(image_string, channels=n_channels)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    resized_image = tf.image.resize_images(image, [size, size])

    if is_training:
        resized_image = data_augment(resized_image, size)

    return resized_image


def train_preprocess(filename, label, random_flip=True):
    """
    Data Augmentation
    """
