from prep_data import prep
from sklearn.model_selection import train_test_split
import random
import pandas as pd

def get_metrics(annotations_df, predictions_df, pred_classes_df, iou_threshold, ignore_classes=False):
    num_frames = len(predictions_df)
    #num_frames = 100
    avg_prec = 0
    avg_rec = 0

    for frame in range(num_frames):
        frame_annotations = annotations_df.iloc[frame,]
        frame_predictions = predictions_df.iloc[frame,]
        pred_classes = pred_classes_df.iloc[frame,]
        tp_count = 0
        fp_count = 0 #predict false, actually true
        
        num_true_annotations = len(frame_annotations['bb_left'])
        num_pred_annotations = len(pred_classes)-1

        for true_annotation_ind in range(num_true_annotations):
            truth_bb = frame_annotations.iloc[true_annotation_ind*4 + 1:true_annotation_ind*4 + 5]
            if true_annotation_ind < 11: true_class = 1
            elif true_annotation_ind < 22: true_class = 2
            else: true_class = 3
            
            # use ground truth annotation with highest iou            
            best_pred_bb = None
            max_iou = 0

            for annotation_ind in range(num_pred_annotations):
                pred_bb = frame_predictions.iloc[annotation_ind*4+1:annotation_ind*4+5]
                pred_class = pred_classes.iloc[annotation_ind]

                if pred_class == true_class or ignore_classes:
                    iou = intersection_size(truth_bb, pred_bb) / union_size(truth_bb, pred_bb) #Intersection over union
                    
                    if iou > max_iou:
                        max_iou = iou
                        best_pred_bb = pred_bb

            if best_pred_bb is None: # No intersection with any truth bounding box
                continue

            #print(max_iou)
            if max_iou > iou_threshold:
                tp_count += 1
        
        # precision = TP / TP + FP = TP / total predictions
        precision = tp_count / num_pred_annotations

        # recall = TP / TP + FN = TP / total true annotations
        recall = tp_count / num_true_annotations

        #if precision > 0 or recall > 0:
        #    print(f"Frame: {frame}")
        #    print(f"Precision: {precision}")
        #    print(f"Recall: {recall}")

        avg_prec += precision / num_frames
        avg_rec += recall / num_frames

        # TODO: Calculate mAP? Or some other useful metric?

    print(f"Avg Precision: {avg_prec}")
    print(f"Avg Recall: {avg_rec}")

def bb_size(bb):
    try:
        return (bb['x_max'] - bb['bb_left']) * (bb['y_max'] - bb['bb_top'])
    except:
        return (bb.iloc[2] - bb.iloc[0]) * (bb.iloc[3] - bb.iloc[1])

def union_size(bb1, bb2):
    return bb_size(bb1) + bb_size(bb2) - intersection_size(bb1, bb2)

def intersection_size(bb1, bb2):
    intersection_left = max(bb1['bb_left'], bb2.iloc[0])
    intersection_top = max(bb1['bb_top'], bb2.iloc[1])
    intersection_right = min(bb1['x_max'], bb2.iloc[2])
    intersection_bottom = min(bb1['y_max'], bb2.iloc[3])

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

    annotations_df = prep(annotations_dir, videos_dir, img_dir, save_frames=False)
    train_df, test_df = train_test_split(annotations_df, test_size=0.1, random_state=42)
    predictions_df = pd.read_csv(predictions_dir+'/boxes.csv')
    pred_labels_df = pd.read_csv(predictions_dir+'/labels.csv')

    get_metrics(test_df, predictions_df, pred_labels_df,  iou_threshold)
    # get_metrics(test_df, predictions_df, pred_labels_df,  iou_threshold, ignore_classes=True)
    # gen_rand_annotations()

if __name__ == "__main__":
    main()
