# Computer-Vision-Final-Project

Colin Chenoweth (cwc63), Thomas Bornhorst (thb34), and Olivia Tchilibou (axt619)

Kaggle Dataset Original Paper:
https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Scott_SoccerTrack_A_Dataset_and_Tracking_Algorithm_for_Soccer_With_Fish-Eye_CVPRW_2022_paper.pdf

Files

prep_data.py: converts dataset into usable Pandas data frame; can extract frames from .mp4 files if needed.

faster_RCNN.py: Main file for this project, creates and trains a torchvision Faster RCNN model.

test.py: Can be used to print predicted and ground truth bounding boxes for test data, or can write predictions to a .csv for further analysis

metrics.py: Extracts and prints precision and recall for a given intersection over the union (IOU) threshold. Can choose to ignore classes or not to get these metrics.

annotate_frames.py: Can overlay bounding boxes onto a frame, and can show the comparison of total ground truth boxes vs predicted boxes.

tracking.py: Used to making an animated tracking .gif of the players and to track possesion of the ball during the game.
