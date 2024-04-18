from prep_data import prep
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

def get_metrics(annotations_df, predictions_df):
    # num_frames = len(predictions_df)
    num_frames = 1

    for frame in range(num_frames):
        frame_annotations = annotations_df.iloc[frame,]
        frame_predictions = predictions_df.iloc[frame,]

        for annotation_ind in range(len(frame_annotations['bb_left'])):
            truth_bb = frame_annotations.iloc[annotation_ind*4 + 1:annotation_ind*4 + 5]
            pred_bb = frame_predictions.iloc[annotation_ind*4 + 1:annotation_ind*4 + 5]

            print(intersection_size(truth_bb, pred_bb) / union_size(truth_bb, pred_bb))

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

def main():
    annotations_dir = './data/top_view/annotations'
    videos_dir = './data/top_view/videos'
    img_dir = './data/top_view/frames'
    predictions_dir = './predictions/annotations'

    annotations_df = prep(annotations_dir, videos_dir, img_dir, save_frames=False, start_vid=0, end_vid=1)
    predictions_df = prep(predictions_dir, videos_dir, img_dir, save_frames=False, start_vid=0, end_vid=1)

    get_metrics(annotations_df, predictions_df)

if __name__ == "__main__":
    main()