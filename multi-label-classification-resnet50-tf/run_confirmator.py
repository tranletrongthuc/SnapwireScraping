import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import redis
from model.resnet import ResNet50
from model.deeplab import DeepLabModel, drawSegment
import logging
from pathlib import Path
from datetime import datetime
import csv
from io import BytesIO

import tensorflow

if tensorflow.version.VERSION[0] == '1':
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_eager_execution()

import numpy as np
import time
import base64
import cv2
from PIL import Image
import io
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import json

# ================================================================================================================================
redis_info = {
    'host' : '127.0.0.1',
    'port' : 6379
}

redis_cli = redis.Redis(host=redis_info['host'], port=redis_info['port'])

tf_JSON_CONFIG = 'config.json'

tf_class_dict = {
    0:'0-OneHand',
    1:'1-TwoHandsAndMore',
    2:'2-NoDelivery'
}

timeline = {
    "Remove_background":0,
    "Load_images_to_model":0,
    "Prepare_model":0,
    "Restore_model_from_weight":0,
    "Run_predict":0,
    "Reinit_model":0,
    "Show_predicted_result":0,
    "Create_video":0,
}

wh_redis_keys = {
    'classification': 'wh_classify_*',
    'visualization':'wh_visual_',
    'test_testimage':'wh_testimage',
    'confirmator_working_state':'wh_confirmator_state',
    'output_dir':'wh_output_dir'
}


video_fps = 7

min_wronghand_count_percentage = 0.3

dtree = None

models_dir = "../../models"

image_testing_mode = False

remove_background_mode = False

classification_history_checked_path = os.path.join(models_dir,'confirmator_classification/classification_history_checked.csv')

processing_count_file_path = os.path.join(models_dir,'processing_count.json')

tf_model_default_dir = os.path.join(models_dir,'confirmator_classification/confirmator_cl_tf_resnet50_fullbody_Dec28_300/checkpoint')

tf_remove_bg_model_dir = os.path.join(models_dir,'remove_bg','mobilenet')

# tf_remove_bg_model_dir = os.path.join(models_dir,'remove_bg','xception')

output_dir = redis_cli.get(wh_redis_keys['output_dir'])

if output_dir is not None:
    log_dir = output_dir.decode("utf-8")
else:
    log_dir = "../../log"

prediction_timeline_path = os.path.join(log_dir,'prediction_timeline.csv')

classification_history_unchecked_path = os.path.join(log_dir,'classification_history_unchecked.csv')

confirmator_log_dir = os.path.join(log_dir, 'confirmator_log')

confirmator_log_default_path = os.path.join(confirmator_log_dir, 'confirmator_date_str.txt')

# Create a custom logger
logger = logging.getLogger(__name__)

layout_path = "../../cpp/cam_info.txt"

# ================================================================================================================================

def init_log(date_string):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    if not os.path.exists(confirmator_log_dir):
        os.mkdir(confirmator_log_dir)

    if not os.path.exists(prediction_timeline_path):
        with open(prediction_timeline_path, mode='w', newline='') as csv_file:
            fieldnames = list(timeline.keys())
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=',')
            writer.writeheader()

    if not os.path.exists(classification_history_unchecked_path):
        with open(classification_history_unchecked_path, mode='w', newline='') as csv2_file:
            fieldnames = list(tf_class_dict.values()) + ['total_found', 'prediction_result', 'file_path']
            writer2 = csv.DictWriter(csv2_file, fieldnames=fieldnames, delimiter=',')
            writer2.writeheader()

    global confirmator_log_default_path
    confirmator_log_path = confirmator_log_default_path.replace('date_str', date_string)
    # Create handlers
    f_handler = logging.FileHandler(confirmator_log_path)
    f_handler.setLevel(logging.INFO)
    f_format = logging.Formatter('%(asctime)s - %(message)s')
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    global logger
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.addHandler(f_handler)

def get_store_ids():
    store_ids = []
    if os.path.exists(layout_path):
        layout_data = []
        with open(layout_path) as file_r:
            layout_data = file_r.readlines()
        store_ids = [line.split(" ")[0] for line in layout_data]
    return store_ids

def init_processing_count_by_date(date_str):
    store_ids = get_store_ids()
    processing_data = {}
    if os.path.exists(processing_count_file_path):
        with open(processing_count_file_path) as json_file_r:
            processing_data = json.load(json_file_r)
    if processing_data.get(date_str) is None:
        processing_data[date_str] = {}
        for store_id in store_ids:
            processing_data[date_str][store_id] = 0
        with open(processing_count_file_path, 'w') as json_file_w:
            json.dump(processing_data, json_file_w)

def update_processing_count_by_date(date_str, store_id):
    data = {}
    if os.path.exists(processing_count_file_path):
        with open(processing_count_file_path) as json_file_r:
            data = json.load(json_file_r)

    if data.get(date_str) is not None:
        if data[date_str].get(store_id) is not None:
            data[date_str][store_id] = int(data[date_str][store_id]) + 1
        else:
            data[date_str][store_id] = 1
    else:
        data[date_str] = {}
        data[date_str][store_id] = 1

    with open(processing_count_file_path, 'w') as json_file_w:
        json.dump(data, json_file_w)

def init_decision_tree():
    global dtree
    dtree = DecisionTreeClassifier()
    classification_history_df = pd.read_csv(classification_history_checked_path)
    features = list(tf_class_dict.values())
    X = classification_history_df[features]
    y = classification_history_df['prediction_result']
    dtree = dtree.fit(X, y)

def load_images_from_redis(prefix):
    list_frame_redis_keys = redis_cli.keys(pattern=prefix)

    # sort list by index
    list_frame_redis_keys = sorted(list_frame_redis_keys, key=lambda x: int(x.decode('utf-8').split("_")[3]))

    images_np = []
    for key in list_frame_redis_keys:
        img_base64 = redis_cli.get(key)

        if img_base64 is not None:
            fetched_img = base64.b64decode(img_base64)
            fetched_img = np.asarray(Image.open(io.BytesIO(fetched_img)))
            fetched_img = cv2.cvtColor(fetched_img, cv2.COLOR_RGB2BGR)
            images_np.append(fetched_img)
    return images_np

# def interpret(filenames, predictions, classes_dict, fullbody_mode, image_testing_mode):
def interpret(filenames, predictions, classes_dict, image_testing_mode):
    # assert len(filenames) == predictions.shape[0]
    predicted_labels = []
    result = [0]*len(tf_class_dict) # 0-OneHand | 1-TwoHandsAndMore | 2-NoDelivery

    for i, file in enumerate(filenames):
        prediction = predictions[i]
        class_index = np.argmax(prediction)
        accuracy = prediction[class_index]
        class_name = classes_dict[class_index]

        if image_testing_mode:
            true_class_index = int(file.split("/")[-1].split("_")[2][0])
            logger.info(f"{file} --> {class_name} with {accuracy * 100}%")
            if class_index == true_class_index:
                result[class_index] += 1
        else:
            # set threshold for class_index if class_index == 0,
            min_index_0 = 0.95
            if class_index == 0:
                if accuracy < min_index_0:
                    class_index = 2

            result[class_index] += 1

            file_name = str(Path(file).name)

            # Rename fullbody cropped files
            renamed_file = f"{Path(file).parent}/{class_index}_{file_name}"
            os.rename(file,renamed_file)
            logger.info(f"{renamed_file} --> {class_name} with {accuracy * 100}%")

        predicted_labels.append(class_index)

    logger.info("TF Model prediction result".center(50, "_"))
    for tf_class_id, tf_class_name in tf_class_dict.items():
        logger.info(f"{tf_class_name}: {result[tf_class_id]}")

    return result, predicted_labels

def wronghand_classify():
    logger.info("START".center(30,"*"))

    global image_testing_mode

    if redis_cli.get(wh_redis_keys['test_testimage']) is not None:
        if redis_cli.get(wh_redis_keys['test_testimage']).decode("utf-8") == "on":
            image_testing_mode = True
    print(f"Testing images: {image_testing_mode}".center(50,"-"))

    confirming_count = 0

    tf.compat.v1.reset_default_graph()
    tf_classify_model = ResNet50(tf_JSON_CONFIG, len(tf_class_dict))

    if remove_background_mode:
        tf_remove_bg_model = DeepLabModel(tf_remove_bg_model_dir)
    else:
        tf_remove_bg_model = None

    count = 0

    while(True):
        today_str = datetime.now().strftime("%d-%m-%Y")

        # init processing count for today
        init_processing_count_by_date(today_str)

        # get all key from redis according to pattern
        list_cropped_image_keys = redis_cli.keys(pattern=wh_redis_keys['classification'])

        # put working state to Redis
        redis_cli.setex(
            wh_redis_keys['confirmator_working_state'],
            60,
            value=datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        ) # 10min

        if len(list_cropped_image_keys) == 0:
            sleep_time = 5
            print(f'Sleep in {sleep_time} second(s)')
            time.sleep(sleep_time)
            continue

        # init log
        init_log(today_str)


        for cropped_image_key in list_cropped_image_keys:
            # ===============================================================================
            count += 1
            # reset model and tf graph every 50 running times
            if count > 20:
                tf.compat.v1.reset_default_graph()
                tf_classify_model = ResNet50(tf_JSON_CONFIG, len(tf_class_dict))
                count = 0
            # ===============================================================================

            renewed_timeline = timeline.copy()

            confirming_count += 1

            print(f"Start processing keys {cropped_image_key}...")
            start = time.time()
            total_predict_result = [0]*len(tf_class_dict)

            list_cropped_image_infos_bi = redis_cli.get(cropped_image_key)
            list_cropped_image_infos_str = list_cropped_image_infos_bi.decode("utf-8").split('***')
            list_cropped_image_infos = []
            list_cropped_image_paths = []
            store_id = list_cropped_image_infos_str.pop(-1)

            for cropped_image_info_str in list_cropped_image_infos_str :
                if cropped_image_info_str == "":
                    continue
                cropped_image_info = cropped_image_info_str.split(',')

                unpredicted_class_id = int(cropped_image_info[3])
                if unpredicted_class_id == -1:
                    # check existing image
                    if os.path.exists(cropped_image_info[0]):
                        list_cropped_image_paths.append(cropped_image_info[0])
                        list_cropped_image_infos.append(cropped_image_info)
                else:
                    total_predict_result[unpredicted_class_id] += 1

            video_name = cropped_image_key.decode("utf-8").split("_")[-1]
            visualize_images_prefix = wh_redis_keys['visualization'] + video_name + "_*"  # example wh_frame_1609399368668_*

            # check list existing images before loading to model
            if len(list_cropped_image_paths) == 0:
                # for production
                if not image_testing_mode:
                    visualize_images_keys = redis_cli.keys(pattern=visualize_images_prefix)
                    for key in visualize_images_keys:
                        redis_cli.delete(key)
                    redis_cli.delete(cropped_image_key)

                continue

            logger.info(f"Start prediction for key {cropped_image_key}".center(100, "-"))

            # Remove Background
            if remove_background_mode:
                start0 = time.time()
                for cropped_image_path in list_cropped_image_paths:
                    rmbg_start = time.time()
                    jpeg_str = open(cropped_image_path, "rb").read()
                    orignal_im = Image.open(BytesIO(jpeg_str))
                    resized_im, seg_map = tf_remove_bg_model.run(orignal_im)
                    drawSegment(resized_im, seg_map, orignal_im.size, cropped_image_path)
                    print(f"Time taken to evaluate segmentation {cropped_image_path.split('/')[-1]} is : {round(time.time()-rmbg_start,2)}")
                renewed_timeline["Remove_background"] = (round(time.time() - start0, 3))

            # load image to model
            start1 = time.time()
            tf_classify_model.training = False
            tf_classify_model.load_images_to_model(list_cropped_image_paths)
            renewed_timeline["Load_images_to_model"] = (round(time.time() - start1, 3))

            # prepare model
            start2 = time.time()
            tf_classify_model.inference()
            renewed_timeline["Prepare_model"] = (round(time.time() - start2, 3))

            start3 = time.time()
            if not tf_classify_model.is_restored:
                tf_classify_model.restore_weight(weights)
            renewed_timeline["Restore_model_from_weight"] = (round(time.time() - start3, 3))

            # run predict
            start4 = time.time()
            predictions = tf_classify_model.predict(weights, debug=False)
            renewed_timeline["Run_predict"] = (round(time.time() - start4, 3))

            # show predict result
            start6 = time.time()
            batch_predict_result, predicted_labels = interpret(list_cropped_image_paths, predictions, tf_class_dict, image_testing_mode)

            # combine 'batch_predict_result' into 'total_predict_result'
            assert len(total_predict_result) == len(batch_predict_result) # force len(total_predict_result) == len(batch_predict_result)
            for tf_class_id in range(len(tf_class_dict)):
                total_predict_result[tf_class_id] += batch_predict_result[tf_class_id]

            logger.info("Total prediction result".center(50, "_"))
            for tf_class_id, tf_class_name in tf_class_dict.items():
                logger.info(f"{tf_class_name}: {total_predict_result[tf_class_id]}")

            prediction_result = dtree.predict([total_predict_result])
            print(f"Decision tree prediction result: {prediction_result}".center(70, "*"))

            if(prediction_result[0] == 1):
                is_wronghand = True
                logger.info("---> Wronghand Detected !!!!")
                print("---> Wronghand Detected !!!!")
            else:
                is_wronghand = False
                logger.info("---> Next!")
                print("---> Next!")
            logger.info(f"TOTAL PROCESSING TIME: {time.time() - start}s".center(100, "-"))
            print(f"TOTAL PROCESSING TIME: {time.time() - start}s".center(100, "-"))
            renewed_timeline["Show_predicted_result"] = (round(time.time() - start6, 3))

            start7 = time.time()

            if is_wronghand:

                last_slash_pos = list_cropped_image_infos[0][0].rfind("/")
                video_dir = list_cropped_image_infos[0][0][:last_slash_pos].replace("stored", "display").replace(
                    "images", "videos").replace("detected_8", "detected_5")
                visualize_images_np = load_images_from_redis(visualize_images_prefix)

                if len(visualize_images_np) == 0:
                    continue

                frame_width = int(visualize_images_np[0].shape[1])
                frame_height = int(visualize_images_np[0].shape[0])

                # draw wronghand circles
                for info, label_id in zip(list_cropped_image_infos, predicted_labels):
                    if label_id != 0:
                        continue

                    radius = int(info[1])
                    center_point = info[2].split(";")
                    center_point = (int(float(center_point[0])), int(float(center_point[1])))

                    frame_id = int(info[0].split("/")[-1].split("_")[1])
                    cv2.circle(visualize_images_np[frame_id],center_point,radius,(0,0,255),2)

                video_writer = cv2.VideoWriter(
                    f'{video_dir}/{video_name}.mp4',
                    cv2.VideoWriter_fourcc('H', '2', '6', '4'),
                    video_fps,
                    (frame_width, frame_height))

                for np_image in visualize_images_np:
                    video_writer.write(np_image)
                video_writer.release()
                logger.info(f"Video saved {video_dir}/{video_name}.mp4")
                print(f"Video saved {video_dir}/{video_name}.mp4")
            renewed_timeline["Create_video"] = (round(time.time() - start7, 3))

            # store new decision records
            with open(classification_history_unchecked_path, mode='a', newline='') as csv_file:
                fieldnames = list(tf_class_dict.values()) + ['total_found', 'prediction_result','file_path']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=',' )
                if is_wronghand:
                    file_path = f"{video_dir}/{video_name}.mp4"
                else:
                    file_path = ""
                line_value = total_predict_result + [len(list_cropped_image_paths), int(is_wronghand),file_path ]
                line_dict = dict(zip(fieldnames, line_value))
                writer.writerow(line_dict)

            update_processing_count_by_date(date_str=datetime.today().strftime('%d-%m-%Y'), store_id=store_id)


            with open(prediction_timeline_path, mode='a', newline='') as csv2_file:
                fieldnames = list(renewed_timeline.keys())
                writer = csv.DictWriter(csv2_file, fieldnames=fieldnames, delimiter=',' )
                writer.writerow(renewed_timeline)

            # for production
            if not image_testing_mode:
                visualize_images_keys = redis_cli.keys(pattern=visualize_images_prefix)
                for key in visualize_images_keys:
                    redis_cli.delete(key)
                redis_cli.delete(cropped_image_key)

            logger.info("\n\n")

            tf.compat.v1.get_variable_scope().reuse_variables()

        if image_testing_mode:
            print(f"IMAGE TESTING DONE".center(50, "*"))
            break



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", type=str, required=True,
                        help="full path to model checkpoint folder",
                        default=tf_model_default_dir)
    args = parser.parse_args()

    weights = os.path.join(args.model_path, 'model.ckpt')
    today_str = datetime.now().strftime("%d-%m-%Y")
    init_log(today_str)
    init_decision_tree()
    wronghand_classify()
