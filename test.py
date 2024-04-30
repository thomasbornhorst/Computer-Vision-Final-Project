import torch
from torch.utils.data import Dataset, DataLoader
import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms
from torchvision.models.detection import faster_rcnn, fasterrcnn_resnet50_fpn
from torch.optim import SGD
from prep_data import prep
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

def test_model(model, data_loader, device):
    model.to(device)
    
    model.eval()
    total_loss = 0
    
    for idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        preds = model(images)
        print(preds[0])

        if idx >= 10: return

def save_model(model):
    torch.save(model.state_dict(), 'rcnn_model.pth')

def load_model(model):
    model.load_state_dict(torch.load('rcnn_model.pth')) 

def main(load_saved_model = False):
    annotations_dir = './data/top_view/annotations'
    videos_dir = './data/top_view/videos'
    img_dir = './data/top_view/frames'
    annotations_df = prep(annotations_dir, videos_dir, img_dir, save_frames=False)
    num_classes = 4 # 22 players + 1 ball

    train_data, test_data = train_test_split(annotations_df, test_size=0.1, random_state=42)
    test_dataset = SoccerTrackDataset(test_data, img_dir)
    test_data_loader = DataLoader(test_dataset, collate_fn=lambda x: list(zip(*x)))

    model = get_faster_rcnn_model(num_classes)

    if(load_saved_model):
        load_model(model)

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.005)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test_model(model, test_data_loader, device)
    # train_model(model, data_loader, optimizer, device, num_epochs=10)

if __name__ == "__main__":
    main()
