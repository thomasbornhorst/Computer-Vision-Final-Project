from prep_data import prep
import pandas as pd
import cv2
import math
from sklearn.model_selection import train_test_split

def draw_bb(frame_img, bb, pred_class):
    # color from class
    for i in range(4):
        if math.isnan(bb.iloc[i]):
            return frame_img
    
    color = (0, 0, 0)
    if pred_class == 1:
        color = (0, 255, 0)
    elif pred_class == 2:
        color = (0, 0, 255)

    return cv2.rectangle(frame_img, (int(bb.iloc[0]), int(bb.iloc[1])), (int(bb.iloc[2]), int(bb.iloc[3])), color, 2)

def annotate_frames(predictions_df, pred_classes_df, test_df, frame_img_path, new_img_path):
    for frame_ind in range(len(predictions_df)):
        print('on frame ', frame_ind)
        if frame_ind >= 200:
            return

        frame = int(test_df.iloc[frame_ind,]['frame'])
        frame_path = test_df.iloc[frame_ind,]['frame_path']
        
        frame_predictions = predictions_df.iloc[frame,]
        pred_classes = pred_classes_df.iloc[frame,]
        #frame_name = '/frame_' + "{:05d}".format(frame+1) + '.jpg'

        frame_img = cv2.imread(frame_path)

        for annotation_ind in range(len(pred_classes)-1):
            pred_bb = frame_predictions.iloc[annotation_ind*4+1:annotation_ind*4+5]
            pred_class = pred_classes.iloc[annotation_ind]
            frame_img = draw_bb(frame_img, pred_bb, pred_class)
        
        if not cv2.imwrite(new_img_path+frame_path[22:], frame_img):
            raise Exception("Could not write image. Make sure that appropriate frame directory exists.")

def main():
    annotations_dir = './data/top_view/annotations'
    videos_dir = './data/top_view/videos'
    img_dir = './data/top_view/frames'
    predictions_dir = './predictions/annotations'
    viz_results__dir = './predictions/viz_results'

    annotations_df = prep(annotations_dir, videos_dir, img_dir, save_frames=False)
    _, test_df = train_test_split(annotations_df, test_size=0.1, random_state=42)

    predictions_df = pd.read_csv(predictions_dir+'/boxes.csv')
    pred_labels_df = pd.read_csv(predictions_dir+'/labels.csv')
    print('done getting data')
    annotate_frames(predictions_df, pred_labels_df, test_df, img_dir, viz_results__dir)

if __name__ == "__main__":
    main()
