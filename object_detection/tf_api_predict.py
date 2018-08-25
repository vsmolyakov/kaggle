import os
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from matplotlib import pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from tqdm import tqdm

CWD_PATH = '/home/ubuntu/rue/object_detector/open_images_fashion'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'export/frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'openimage_label_map.pbtxt')

IMAGE_SIZE = (12, 8)
PATH_TO_TEST_IMAGES_DIR = '/home/ubuntu/rue/object_detector/open_images_fashion/challenge2018_test'
TEST_IMAGE_PATHS = os.listdir(PATH_TO_TEST_IMAGES_DIR) 

NUM_CLASSES = 493 

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
class_names = pd.read_csv(os.path.join(CWD_PATH, 'class_names.csv'), header=None)

l2n_dict = {}
n2l_dict = {}
for i in range(class_names.shape[0]):
    label = class_names.iloc[i,0]
    name = class_names.iloc[i,1]
    l2n_dict[label] = name
    n2l_dict[name] = label
#end for 


def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    """
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    """

    return boxes, scores, classes 

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

#Load a frozen TF model 
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

TOP_K = 3
SCORE_THRESHOLD = 0.15
submission_df = pd.DataFrame(columns=['ImageId', 'PredictionString'])

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        for idx, image_path in tqdm(enumerate(TEST_IMAGE_PATHS)):
            image = Image.open(PATH_TO_TEST_IMAGES_DIR + '/' + image_path)
            image_np = load_image_into_numpy_array(image)
            image_name = image_path.split('/')[-1].split('.')[0] 
            image_boxes, image_scores, image_classes = detect_objects(image_np, sess, detection_graph)
            image_scores_top = image_scores.flatten()[:TOP_K]
            image_classes_top = image_classes.flatten()[:TOP_K]
            prediction_str = ""
            for i in range(TOP_K):
                if (image_scores_top[i] > SCORE_THRESHOLD):
                    image_object_label = category_index[image_classes_top[i]]['name']
                    y_min, x_min, y_max, x_max = image_boxes[0,i,:]
                    #print(image_object_label)
                    #print(n2l_dict[image_object_label])
                    #print(image_object_box)
                    prediction_str += n2l_dict[image_object_label] + " " + str(round(image_scores_top[i], 2)) + " " + str(round(x_min, 4)) + " " + str(round(y_min, 4)) + " " + str(round(x_max, 4)) + " " + str(round(y_max, 4)) + " " 
            #end for
            print("{},{}".format(image_name, prediction_str))
            submission_df.loc[idx,'ImageId'] = image_name
            submission_df.loc[idx,'PredictionString'] = prediction_str

submission_df.to_csv("./ssd_76880_top3_t015_with_scores_ordered.csv", index=False)

