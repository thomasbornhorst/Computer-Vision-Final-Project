import torch
from torch.utils.data import Dataset, DataLoader
import os
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.models.detection import faster_rcnn, fasterrcnn_resnet50_fpn
from prep_data import prep
import argparse
from sklearn.model_selection import train_test_split

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
        labels = [1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,3]
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
    model = fasterrcnn_resnet50_fpn()
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one for the specified number of classes
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    return model

def train_model(model, data_loader, optimizer, device, num_epochs=10):
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for iter, (images, targets) in enumerate(data_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

            if iter % 50 == 0:
                print(f"Epoch {epoch}, iter {iter}, loss: {total_loss / (iter + 1)}")
                save_model(model)
        
        print(f"Epoch #{epoch+1} Loss: {total_loss / len(data_loader)}")
        save_model(model)

def main(load_saved_model = False):
    annotations_dir = './data/top_view/annotations'
    videos_dir = './data/top_view/videos'
    img_dir = './data/top_view/frames'
    annotations_df = prep(annotations_dir, videos_dir, img_dir, save_frames=False)
    num_classes = 4 # 22 players + 1 ball

    train_data, test_data = train_test_split(annotations_df, test_size=0.1, random_state=42)
    dataset = SoccerTrackDataset(train_data, img_dir)
    data_loader = DataLoader(dataset, batch_size=8, collate_fn=lambda x: list(zip(*x)))

    model = get_faster_rcnn_model(num_classes)

    if(load_saved_model):
        load_model(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.005)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_model(model, data_loader, optimizer, device, num_epochs=2)

def save_model(model):
    torch.save(model.state_dict(), 'rcnn_model.pth')

def load_model(model):
    model.load_state_dict(torch.load('rcnn_model.pth')) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='FastRCNN', description='Trains faster RCNN on frame data')
    parser.add_argument('-l', action='store_true')
    args = parser.parse_args()

    main(load_saved_model=args.l)

