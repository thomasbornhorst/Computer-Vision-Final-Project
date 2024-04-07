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
import tensorflow as TF
from prep_data import prep

class SoccerTrackDataset(Dataset):
    def __init__(self, annotations, img_dir, transforms=None):
        self.annotations = annotations
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = row['frame_path']
        image = read_image(img_path).float() / 255.0  # Normalize to [0, 1]
        
        # Increment player idID by 1 to reserve 0 for the background
        boxes = torch.tensor([row['bb_left'], row['bb_top'], row['x_max'], row['y_max']], dtype=torch.float32)
        boxes = boxes.T
        labels = range(1,24)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])

        if self.transforms:
            image = self.transforms(image)
        
        return image, target

def get_faster_rcnn_model(num_classes):
    """
    Load a pre-trained Faster R-CNN model and replace the classifier head with one
    that has `num_classes`, accounting for the background and player IDs.

    Parameters:
    - num_classes (int): The total number of classes including the background.

    Returns:
    - model (FasterRCNN): A Faster R-CNN model adjusted for the specified number of classes.
    """
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one for the specified number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def train_model(model, data_loader, optimizer, device, num_epochs=10):
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
        
        print(f"Epoch #{epoch+1} Loss: {total_loss / len(data_loader)}")

def main():
    annotations_dir = '.\\data\\top_view\\annotations'
    videos_dir = '.\\data\\top_view\\videos'
    img_dir = '.\\data\\top_view\\frames'
    annotations_df = prep(annotations_dir, videos_dir, img_dir, save_frames=False)
    num_classes = 23 # 22 players + 1 ball

    dataset = SoccerTrackDataset(annotations_df, img_dir)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: list(zip(*x)))

    model = get_faster_rcnn_model(num_classes)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.005)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_model(model, data_loader, optimizer, device, num_epochs=10)

if __name__ == "__main__":
    main()