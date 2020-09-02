from keras.models import load_model
import numpy as np


def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors


def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [class_name.strip() for class_name in class_names]
    return class_names


def read_trained_model(model_path, anchors_path, classes_path):
    model = load_model(model_path, compile=False)
    class_names = read_classes(classes_path)
    anchors = read_anchors(anchors_path)

    # Verify model, anchors, and classes are compatible
    num_classes = len(class_names)
    num_anchors = len(anchors)
    # Assumes dim ordering is channel last
    model_output_channels = model.layers[-1].output_shape[-1]
    assert model_output_channels == num_anchors * (num_classes + 5)

    print('YOLO model loaded successfully!')
    return model, class_names, anchors
