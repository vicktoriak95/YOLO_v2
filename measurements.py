import numpy as np
import os
import re
import shutil
import time

MAP_RELATIVE_PATH = r".\mAP"
MAP_GT_FOLDER_PATH = os.path.join(MAP_RELATIVE_PATH, r"input\ground-truth")
MAP_DETECTIONS_FOLDER_PATH = os.path.join(MAP_RELATIVE_PATH, r"input\detection-results")
MAP_OUTPUT_OUTPUT_PATH = os.path.join(MAP_RELATIVE_PATH, r"output")
MAP_OUTPUT_FILE_PATH = os.path.join(MAP_RELATIVE_PATH, r"output\output.txt")


def write_gt_file_for_mAP(gt_boxes, image_id):
    with open(os.path.join(MAP_GT_FOLDER_PATH, image_id + ".txt"), "w") as f:
        for box in gt_boxes:
            f.write("{class_name} {left} {top} {right} {bottom}\n".format(class_name="person", left=box[0], top=box[1], right=box[2], bottom=box[3]))


def write_predictions_file_for_mAP(out_boxes, out_classes, out_scores, image_name):
    out_boxes = np.array([[box[1], box[0], box[3], box[2]] for box in out_boxes])
    people_indexes = [i for i in range(len(out_classes)) if out_classes[i] == 0]
    out_boxes, out_classes, out_scores = out_boxes[people_indexes], out_classes[people_indexes], out_scores[people_indexes]
    with open(os.path.join(MAP_DETECTIONS_FOLDER_PATH, image_name.split(".")[0] + ".txt"), "w") as f:
        for c, score, box in zip(out_classes, out_scores, out_boxes):
            f.write("{class_name} {confidence} {left} {top} {right} {bottom}\n".format(class_name="person", confidence=score, left=box[0], top=box[1], right=box[2], bottom=box[3]))


def clean_measurements():
    # Clean input
    for file_name in os.listdir(MAP_GT_FOLDER_PATH):
        file_path = os.path.join(MAP_GT_FOLDER_PATH, file_name)
        os.remove(file_path)
    for file_name in os.listdir(MAP_DETECTIONS_FOLDER_PATH):
        file_path = os.path.join(MAP_DETECTIONS_FOLDER_PATH, file_name)
        os.remove(file_path)
    # Clean output
    if os.path.exists(MAP_OUTPUT_OUTPUT_PATH):  # if it exist already
        shutil.rmtree(MAP_OUTPUT_OUTPUT_PATH)
        time.sleep(1)


def calc_measurements():
    # Run mAP main.py
    os.system("cd {map} && python main.py --no-plot --quiet".format(map=MAP_RELATIVE_PATH))

    # Extract the measurements
    with open(MAP_OUTPUT_FILE_PATH) as f:
        estimation_data = f.read()
    c = re.compile(r"mAP = ([0-9]+.[0-9]+)%")
    mAP = float(c.findall(estimation_data)[0])
    c = re.compile(r" Precision: (\[.*\])")
    precision = float(eval(c.findall(estimation_data)[0])[-1])
    c = re.compile(r" Recall :(\[.*\])")
    recall = float(eval(c.findall(estimation_data)[0])[-1])
    return mAP, precision, recall
