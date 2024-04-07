import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.optim import SGD
from torchvision.transforms import Compose, ToTensor, Normalize
from sklearn.model_selection import train_test_split


# bb_left and bb_top are the top left corner, and bb_width and bb_height are the dimensions of the bounding box
# x_max would be bb_left + bb_width
# y_max would be bb_top + bb_height
# x_min = bb_left
# y_min = bb_top
# Aim to track the player id in each frame

def load_annotations(csv_path):
    data = pd.read_csv(csv_path, header=None)
    data.columns = data.iloc[2]
    data = data.drop([0, 1, 2, 3])
    data = data.rename({'Attributes':'frame'},axis=1)
    return data.astype(float)

def extract_frames(video_path, output_dir, frame_numbers, past_frames):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    success, frame_idx = True, 1
    while success:
        success, frame = cap.read()
        if frame_idx in frame_numbers and success:
            frame_path = os.path.join(output_dir, f"frame_{(frame_idx+past_frames):04d}.jpg")
            cv2.imwrite(frame_path, frame)
        frame_idx += 1
    cap.release()

def get_frame_path(frame_number, output_dir):
    return os.path.join(output_dir, f"frame_{frame_number:04f}.jpg")

def preprocess_annotations(annotations, output_dir):
    temp1 = annotations['bb_left'].add(annotations['bb_width'].rename({'bb_width':'bb_left'},axis=1),fill_value=0)
    annotations['bb_width'] = temp1
    temp2 = annotations['bb_top'].add(annotations['bb_height'].rename({'bb_height':'bb_top'},axis=1),fill_value=0)
    annotations['bb_height'] = temp2
    annotations = annotations.rename({'bb_width':'x_max', 'bb_height':'y_max'}, axis=1)
    annotations['frame_path'] = annotations['frame'].apply(lambda x: get_frame_path(x, output_dir))
    return annotations

def split_dataset(annotations, val_size=0.2, test_size=0.1):
    initial_train_and_val_size = 1 - test_size
    train_val_annots, test_annots = train_test_split(annotations, test_size=test_size)
    adjusted_val_size = val_size / initial_train_and_val_size
    train_annots, val_annots = train_test_split(train_val_annots, test_size=adjusted_val_size)
    return train_annots, val_annots, test_annots

def prep(annotations_dir, videos_dir, frames_output_dir, save_frames=False):
    all_annotations = []
    past_frames = 0
    
    for i in range(60):  # Assuming 60 pairs of videos and CSV files (top view dataset)
        csv_path = os.path.join(annotations_dir, f"D_20220220_1_{i*30:04d}_{(i+1)*30:04d}.csv")
        video_path = os.path.join(videos_dir, f"D_20220220_1_{i*30:04d}_{(i+1)*30:04d}.mp4")
        
        annotations = load_annotations(csv_path)
        frame_numbers = sorted(annotations['frame'].unique())
        
        if save_frames:
            extract_frames(video_path, frames_output_dir, frame_numbers, past_frames)
        
        preprocessed_annotations = preprocess_annotations(annotations, frames_output_dir)
        all_annotations.append(preprocessed_annotations)
        past_frames += len(frame_numbers)
    
    all_annotations_df = pd.concat(all_annotations, ignore_index=True)
    
    train_annots, val_annots, test_annots = split_dataset(all_annotations_df, val_size=0.2, test_size=0.1)

    #  use train_annots, val_annots, and test_annots for training, validating, and testing the model
    print("Training annotations:", len(train_annots))
    print("Validation annotations:", len(val_annots))
    print("Testing annotations:", len(test_annots))


annotations_dir = '.\\data\\top_view\\annotations'
videos_dir = '.\\data\\top_view\\videos'
frames_output_dir = '.\\data\\top_view\\frames'
prep(annotations_dir, videos_dir, frames_output_dir, save_frames=False)
