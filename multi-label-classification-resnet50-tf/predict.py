import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
from model.resnet import ResNet50
import logging
from pathlib import Path


import tensorflow

if tensorflow.version.VERSION[0] == '1':
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_eager_execution()

import numpy as np
import time

tf_classify_model_path = ""
tf_JSON_CONFIG = 'config.json'
tf_class_dict = {
    0:'0-OneHand',
    1:'1-TwoHandsAndMore',
    2:'2-NoDelivery'
}
tf_classify_model = None
video_fps = 7


def interpret(filenames, predictions, classes_dict):
    assert len(filenames) == predictions.shape[0]
    predicted_labels = []
    result = [] # 0-OneHand | 1-TwoHandsAndMore | 2-NoDelivery
    for tf_class_id in range(len(tf_class_dict)):
        result.append(0)

    for i, file in enumerate(filenames):
        prediction = predictions[i]
        class_index = np.argmax(prediction)
        accuracy = prediction[class_index]

        if class_index == 0:
            if accuracy < 0.95:
                # get the 2nd peak
                prediction_temp = list(prediction)
                prediction_temp[class_index] = -1
                class_index = np.argmax(prediction_temp)
                accuracy = prediction_temp[class_index]

        class_name = classes_dict[class_index]

        result[class_index] += 1

        # Rename fullbody cropped files
        renamed_file = f"{Path(file).parent}/{class_index}_{Path(file).name}"
        os.rename(file,renamed_file)
        logging.info(f"{renamed_file} --> {class_name} with {accuracy * 100}%")

        predicted_labels.append(class_index)

    for tf_class_id, tf_class_name in tf_class_dict.items():
        logging.info(f"{tf_class_name}: {result[tf_class_id]}")

    return result, predicted_labels


def wronghand_classify(cropped_img_dir):
    # get all key from redis according to pattern
    list_cropped_image_paths = [os.path.join(cropped_img_dir,name) for name in os.listdir(cropped_img_dir)]

    # load image to model
    tf_classify_model.load_images_to_model(list_cropped_image_paths)

    # prepare model
    tf_classify_model.inference()
    if not tf_classify_model.is_restored:
        tf_classify_model.restore_weight(weights)

    # run predict
    predictions = tf_classify_model.predict(debug=False)
    tf.get_variable_scope().reuse_variables()

    # show predict result
    interpret(list_cropped_image_paths, predictions, tf_class_dict)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", type=str, required=True,
                        help="full path to model checkpoint folder",
                        default="../../models/confirmator_classification/confirmator_cl_tf_resnet50_fullbody_Dec28_300/checkpoint")
    parser.add_argument("-log", "--log-path", type=str,
                        help="path to log file to write.",
                        default='../../output/confirmator.txt')
    parser.add_argument("-img", "--cropped-img-dir", type=str,
                        help="cropped image folder.",
                        default='../../output')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s',
                        filename=args.log_path,
                        filemode='w',
                        level=logging.DEBUG)
    weights = os.path.join(args.model_path, 'model.ckpt')
    tf_classify_model = ResNet50(tf_JSON_CONFIG, len(tf_class_dict))

    wronghand_classify(args.cropped_img_dir)
