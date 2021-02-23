import tensorflow

if tensorflow.version.VERSION[0] == '1':
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf

    tf.compat.v1.disable_eager_execution()


from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    # INPUT_SIZE = 513
    INPUT_SIZE = 512
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        graph_def = tf.GraphDef.FromString(open(tarball_path + "/frozen_inference_graph.pb", "rb").read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """

        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

        # decrease brightness
        enhancer = ImageEnhance.Brightness(resized_image)
        processed_im = enhancer.enhance(0.7)

        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(processed_im)]})
        seg_map = batch_seg_map[0]

        return resized_image, seg_map

def drawSegment(baseImg, matImg, org_size, outputFilePath):
    width, height = baseImg.size
    dummyImg = np.zeros([height, width, 4], dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            color = matImg[y, x]
            (r, g, b) = baseImg.getpixel((x, y))
            if color == 0:
                dummyImg[y, x, 3] = 0
            else:
                dummyImg[y, x] = [r, g, b, 255]
    dummyImg = cv2.cvtColor(dummyImg,cv2.COLOR_RGB2BGR)
    dummyImg = cv2.resize(dummyImg,org_size)
    cv2.imwrite(outputFilePath, dummyImg)