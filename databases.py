import json
import numpy as np
import re
import os

KNOWN_DBS = {"crowd": {"input_path": r".\databases\CrowdHuman_val\Images",
                       "output_path": r".\databases\CrowdHuman_val\Images\out",
                       "gt_path": r".\databases\CrowdHuman_val\annotation_val.odgt"},
             "inria": {"input_path": r".\databases\INRIAPerson\images",
                       "output_path": r".\databases\INRIAPerson\images\out",
                       "gt_path": r".\databases\INRIAPerson\annotations"}}


######################## CROWD ########################
def read_gt_crowd():
    with open(KNOWN_DBS["crowd"]["gt_path"]) as f:
        gt_crowd_data = f.read()
    gt_crowd = [json.loads(gt_crowd_line) for gt_crowd_line in gt_crowd_data.splitlines()]
    return gt_crowd


def get_crowd_image_gt(image_id):
    gt_crowd = read_gt_crowd()
    for image in gt_crowd:
        if image["ID"] == image_id:
            return image
    raise KeyError("ID {id} not found is crowd database".format(image_id))


def get_crowd_image_gt_boxes(image_id):
    image_gt = get_crowd_image_gt(image_id)
    visible_people_boxes = [box["vbox"] for box in image_gt["gtboxes"] if box["tag"] == "person"]
    # return (x_min, y_min, x_max, y_max) instead of (x_min, y_min, width, height)
    gt_boxes = np.array([np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]]) for box in visible_people_boxes])
    return gt_boxes


KNOWN_DBS["crowd"]["get_image_gt_boxes"] = get_crowd_image_gt_boxes


######################## INRIAPerson ########################
def get_inria_image_gt_boxes(image_id):
    with open(os.path.join(KNOWN_DBS["inria"]["gt_path"], image_id + ".txt")) as f:
        image_gt_data = f.read()
    c = re.compile(
        r"Bounding box for object \d+ \"PASperson\" \(Xmin, Ymin\) - \(Xmax, Ymax\) : \((\d+), (\d+)\) - \((\d+), (\d+)\)")
    gt_boxes = c.findall(image_gt_data)
    gt_boxes = [[int(n) for n in box] for box in gt_boxes]
    return gt_boxes


KNOWN_DBS["inria"]["get_image_gt_boxes"] = get_inria_image_gt_boxes


######################## General ########################
def get_image_gt_boxes(image_id, db):
    return KNOWN_DBS[db]["get_image_gt_boxes"](image_id)


def get_io_paths_from_db(db):
    if db is None:
        input_path = 'images/'
        output_path = 'images/out'
    else:
        input_path = KNOWN_DBS[db]["input_path"]
        output_path = KNOWN_DBS[db]["output_path"]
    # create output path if needed
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    return input_path, output_path
