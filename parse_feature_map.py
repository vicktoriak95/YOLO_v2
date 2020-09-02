import tensorflow as tf
from keras import backend as K
import numpy as np
from PIL import Image


def prepare_image_for_network(image, model_image_size):
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image_data


def get_predictions_from_feature_map(feats, anchors, num_classes):
    """Extract all predictions from feature map.

    :param feats: the feature map - the output of the convolutional network
    :param anchors: the model anchors
    :param num_classes: number of classes the model can predict
    :return: tuple:
        box_xy - for every grid cell - list of (x_center, y_center) of a predicted object, of size len(anchors)
        box_wh - for every grid cell - list of (width, height) of a predicted object, of size len(anchors)
        box_confidence - for every grid cell - list of confidences of a predicted object, of size len(anchors)
        box_class_probs - for every grid cell - list of probability list of a predicted object, of size len(anchors)
    """
    # calc anchors_tensor
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])

    # calc conv_index
    conv_dims = K.shape(feats)[1:3]  # assuming channels last
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])
    conv_width_index = K.tile(K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))

    # reshape feature map
    feats = K.reshape(feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
    # calc conv_dims
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

    # extract predictions from feature map
    box_xy = K.sigmoid(feats[..., :2])
    box_wh = K.exp(feats[..., 2:4])
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.softmax(feats[..., 5:])

    # box_xy relative to whole picture (not to grid cell)
    box_xy = (box_xy + conv_index) / conv_dims
    # box_wh relative to whole picture (not to anchors)
    box_wh = box_wh * anchors_tensor / conv_dims

    return box_xy, box_wh, box_confidence, box_class_probs


def boxes_to_corners(box_xy, box_wh):
    """
    :param box_xy: 1D col array of (x, y) centers of boxes
    :param box_wh: 1D col array of (w, h) of boxes
    :return: list of boxes in the format (y_min, x_min, y_max, x_max)
    """
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return K.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])


def filter_boxes(boxes, box_confidence, box_class_probs, threshold=.6):
    """
    Filter YOLO boxes based on object and class confidence.
    :param boxes: num_anchors * boxes in the format (y_min, x_min, y_max, x_max)
    :param box_confidence: num_anchors * 1 tensor - confidence that the box contains an object
    :param box_class_probs: num_anchors * num_classes tensor - for each box probability array for each class
    :param threshold: above this score an object is considered detected
    :return: boxes, scores, classes of detected objects with score >= threshold
    """
    # box_scores[box][class] = probability that *box* conaints an object & object is *class*
    box_scores = box_confidence * box_class_probs
    # box_classes[box] = index class that *box* most likely belongs to
    box_classes = K.argmax(box_scores, axis=-1)
    # box_class_scores[box] = probability that *box* belongs to the most probable class
    box_class_scores = K.max(box_scores, axis=-1)
    # prediction_mask[box] = True / False - is *box* score high enough to determine it detected an object
    prediction_mask = box_class_scores >= threshold

    boxes = tf.boolean_mask(boxes, prediction_mask)
    scores = tf.boolean_mask(box_class_scores, prediction_mask)
    classes = tf.boolean_mask(box_classes, prediction_mask)
    return boxes, scores, classes


def get_detected_boxes_from_predictions(yolo_output_predictions, image_shape, max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """

    :param yolo_output_predictions: box_xy, box_wh, box_confidence, box_class_probs
    :param image_shape: the shape of the input image
    :param max_boxes: the maximum number of boxes to be selected by non max suppression.
    :param score_threshold: the certainty above which an object will be considered as detected
    :param iou_threshold: the threshold for deciding whether boxes overlap too much with respect to IOU.
    :return: detected_boxes, detected_scores, detected_classes
    """
    box_xy, box_wh, box_confidence, box_class_probs = yolo_output_predictions
    # convert boxes to (y_min, x_min, y_max, x_max) format from (x_center, y_center, width, height) format
    boxes = boxes_to_corners(box_xy, box_wh)
    # eliminate objects with score < score_threshold
    boxes, scores, classes = filter_boxes(boxes, box_confidence, box_class_probs, threshold=score_threshold)

    # Scale boxes back to original image shape
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims

    # Remove overlapping boxes assuming they discover the same object
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.compat.v1.variables_initializer([max_boxes_tensor]))
    nms_index = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)
    detected_boxes = K.gather(boxes, nms_index)
    detected_scores = K.gather(scores, nms_index)
    detected_classes = K.gather(classes, nms_index)
    return detected_boxes, detected_scores, detected_classes
