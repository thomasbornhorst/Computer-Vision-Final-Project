from prep_data import prep
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import random

def get_metrics(annotations_df, predictions_df, iou_threshold):
    # num_frames = len(predictions_df)
    num_frames = 1

    for frame in range(num_frames):
        frame_annotations = annotations_df.iloc[frame,]
        frame_predictions = predictions_df.iloc[frame,]
        tp_count = 0
        fp_count = 0 #predict false, actually true

        num_true_annotations = len(frame_annotations['bb_left'])
        num_pred_annotations = len(frame_predictions['bb_left'])

        for annotation_ind in range(num_pred_annotations):
            pred_bb = frame_predictions.iloc[annotation_ind*4 + 1:annotation_ind*4 + 5]

            # use ground truth annotation with highest iou            
            best_truth_bb = None
            max_iou = 0

            for true_annotation_ind in range(num_true_annotations):
                truth_bb = frame_annotations.iloc[true_annotation_ind*4 + 1:true_annotation_ind*4 + 5]
                iou = intersection_size(truth_bb, pred_bb) / union_size(truth_bb, pred_bb) #Intersection over union

                if iou > max_iou:
                    max_iou = iou
                    best_truth_bb = truth_bb

            if best_truth_bb is None: # No intersection with any truth bounding box
                continue

            # TODO: Check if correctly picked team 1 / team 2 / ball
            if max_iou > iou_threshold:
                tp_count += 1
        
        # precision = TP / TP + FP = TP / total predictions
        precision = tp_count / num_pred_annotations

        # recall = TP / TP + FN = TP / total true annotations
        recall = tp_count / num_true_annotations

        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

        # TODO: Calculate mAP? Or some other useful metric?

def bb_size(bb):
    return (bb['x_max'] - bb['bb_left']) * (bb['y_max'] - bb['bb_top'])

def union_size(bb1, bb2):
    return bb_size(bb1) + bb_size(bb2) - intersection_size(bb1, bb2)

def intersection_size(bb1, bb2):
    intersection_left = max(bb1['bb_left'], bb2['bb_left'])
    intersection_top = max(bb1['bb_top'], bb2['bb_top'])
    intersection_right = min(bb1['x_max'], bb2['x_max'])
    intersection_bottom = min(bb1['y_max'], bb2['y_max'])

    intersection_width = intersection_right - intersection_left
    intersection_height = intersection_bottom - intersection_top

    if intersection_width > 0 and intersection_height > 0:
        return intersection_width * intersection_height
    
    return 0

def gen_rand_annotations():
    x_max = 3840
    y_max = 2160
    player_height = 39 # Players 39 x 39
    # ball_height = 10 # Ball 10 x 10

    for _ in range(23): # 22 Players + Ball
        bb_left = random.randint(0, x_max - player_height)
        bb_right = bb_left + player_height
        bb_top = random.randint(0, y_max - player_height)
        bb_bot = bb_top + player_height

def main():
    annotations_dir = './data/top_view/annotations'
    videos_dir = './data/top_view/videos'
    img_dir = './data/top_view/frames'
    predictions_dir = './predictions/annotations'
    iou_threshold = 0.5

    annotations_df = prep(annotations_dir, videos_dir, img_dir, save_frames=False, start_vid=0, end_vid=1)
    predictions_df = prep(predictions_dir, videos_dir, img_dir, save_frames=False, start_vid=0, end_vid=1)

    get_metrics(annotations_df, predictions_df, iou_threshold)
    # gen_rand_annotations()

if __name__ == "__main__":
    main()