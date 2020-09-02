from utils import suppress_irrelevant_warnings
suppress_irrelevant_warnings()

import os
from keras import backend as K
import argparse
from PIL import Image
import time

from load_model import read_trained_model
from parse_feature_map import get_predictions_from_feature_map, get_detected_boxes_from_predictions, prepare_image_for_network
from databases import get_image_gt_boxes, get_io_paths_from_db, KNOWN_DBS
from measurements import clean_measurements, write_gt_file_for_mAP, write_predictions_file_for_mAP, calc_measurements
from graphics import draw_object_on_image, RED, GREEN


def run_yolo_v2(model_path, anchors_path, classes_path, db, no_draw, score_threshold, iou_threshold):
    print("\nStarting Yolo_v2. Enjoy!")
    # Set input & output paths according to the selected db
    input_path, output_path = get_io_paths_from_db(db)

    sess = K.get_session()
    # Read the model, the classes it can detect and its anchors from the files.
    yolo_model, class_names, anchors = read_trained_model(model_path, anchors_path, classes_path)
    # Generate output tensor targets for filtered bounding boxes.
    yolo_output_predictions = get_predictions_from_feature_map(yolo_model.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    detected_boxes, detected_scores, detected_classes = \
        get_detected_boxes_from_predictions(yolo_output_predictions,
                                            input_image_shape,
                                            score_threshold=score_threshold,
                                            iou_threshold=iou_threshold)

    times = []
    clean_measurements()
    for image_name in os.listdir(input_path):
        # Not a file -> not an image -> not interested
        if not os.path.isfile(os.path.join(input_path, image_name)):
            continue
        start_time = time.time()

        # Read the input image
        image = Image.open(os.path.join(input_path, image_name))
        # Assumes dim ordering is channel last
        model_image_size = yolo_model.layers[0].input_shape[1:3]
        image_data = prepare_image_for_network(image, model_image_size)

        # Run the image through the network
        out_boxes, out_scores, out_classes = sess.run(
            [detected_boxes, detected_scores, detected_classes],
            feed_dict={
                yolo_model.input: image_data,
                input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        print('\nFound {n} objects for {image_name}:'.format(n=len(out_boxes), image_name=image_name))

        # Write object predictions and ground-truth for future mAP calculation
        # if db is None - there is no ground-truth and therefore there is no mAP calculation
        if db is not None:
            image_id = image_name.split(".")[0]
            gt_boxes = get_image_gt_boxes(image_id, db)
            write_gt_file_for_mAP(gt_boxes, image_id)
            write_predictions_file_for_mAP(out_boxes, out_classes, out_scores, image_name)

        # Save an output image with the ground-truth objects and the predicted objects
        if not no_draw:
            # Iterate over the ground-truth objects and add them to the output image
            # if db is None - there is no ground-truth to draw on the image
            if db is not None:
                for gt_box in gt_boxes:
                    draw_object_on_image(image, [gt_box[1], gt_box[0], gt_box[3], gt_box[2]], 1, "person", GREEN)
            # Iterate over the objects detected and add them to the output image
            for box, score, class_id in zip(out_boxes, out_scores, out_classes):
                predicted_class = class_names[class_id]
                # draw only people
                if predicted_class != "person":
                    continue
                draw_object_on_image(image, box, score, predicted_class, RED)
            image.save(os.path.join(output_path, image_name), quality=90)

        # Print processing time for the image
        end_time = time.time()
        times.append(end_time - start_time)
        print("Image processing took: {t} seconds".format(t=(end_time - start_time)))

    sess.close()
    print("\nImage processing took on avg: {t} seconds".format(t=(sum(times)/len(times))))
    # Calc mAP
    # if db is None - there is no ground-truth and therefore there is no mAP calculation
    if db is not None:
        mAP, precision, recall = calc_measurements()
        print("\n\nThe network measurements for database {db} are as follows:\n"
              "mAP = {mAP}\n"
              "Precision = {precision}\n"
              "Recall = {recall}\n".format(db=db, mAP=mAP, precision=precision, recall=recall))


def parse_args():
    parser = argparse.ArgumentParser(description='Run a YOLO_v2 detection model on images in a given directory..')
    parser.add_argument('--db', help='The database used as input images for the model testing. '
                                     'Defaults to None - in that case the images/ folder will be used. ',
                        default=None)
    parser.add_argument('--model_path', help='Path to h5 model file containing body of a YOLO_v2 model, '
                                             'defaults to model_data/yolo.h5',
                        default='model_data/yolo.h5')
    parser.add_argument('--anchors_path', help='Path to anchors file, '
                                               'defaults to model_data/yolo_anchors.txt',
                        default='model_data/yolo_anchors.txt')
    parser.add_argument('--classes_path', help='Path to classes file, '
                                               'defaults to model_data/coco_classes.txt',
                        default='model_data/coco_classes.txt')
    parser.add_argument('--no-draw', help='If set - images with the detected objects will not be drawn. '
                                          'In that case - the script only calculates measurements',
                        action='store_true')
    parser.add_argument('--score_threshold', type=float, help='threshold for bounding box scores, default .3',
                        default=.3)
    parser.add_argument('--iou_threshold', type=float, help='threshold for non max suppression IOU, default .5',
                        default=.5)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # validate model given in the correct format
    assert args.model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
    # validate db in from white list of supported dbs
    assert args.db is None or args.db in KNOWN_DBS

    run_yolo_v2(args.model_path, args.anchors_path, args.classes_path, args.db, args.no_draw, args.score_threshold, args.iou_threshold)


if __name__ == '__main__':
    main()
