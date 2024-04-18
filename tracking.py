from prep_data import prep
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

def make_tracking_gif(df):
    fig, ax = plt.subplots()

    team1 = ax.scatter(df.iloc[0, :11]['x_center'].to_list(), df.iloc[0, 23:34]['y_center'].to_list(), color='blue', label='Team 1')
    team2 = ax.scatter(df.iloc[0, 11:22]['x_center'].to_list(), df.iloc[0, 34:45]['y_center'].to_list(), color='red', label='Team 2')
    ball = ax.scatter(df.iloc[0, 22:23]['x_center'], df.iloc[0, 45:46]['y_center'], color='black', label='Ball', facecolors='none')

    ax.set_xlim(0,3840)
    ax.set_ylim(0,2160)
    ax.set_title('Soccer Field')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.legend()

    def update(frame):
        team1_positions = np.array([df.iloc[frame, :11]['x_center'].to_numpy(), df.iloc[frame, 23:34]['y_center'].to_numpy()])
        team2_positions = np.array([df.iloc[frame, 11:22]['x_center'].to_numpy(), df.iloc[frame, 34:45]['y_center'].to_numpy()])
        ball_position = np.array([df.iloc[frame, 22:23]['x_center'], df.iloc[frame, 45:46]['y_center']])

        team1.set_offsets(team1_positions.T)
        team2.set_offsets(team2_positions.T)
        ball.set_offsets(ball_position)

        return team1, team2, ball

    ani = FuncAnimation(fig, update, frames=len(df), interval=1, blit=True)
    ani.save('tracking_animation.gif', fps=30)
    

def centers_only(annotations_df):
    temp1 = annotations_df['bb_left'].add(annotations_df['x_max'].rename({'x_max':'bb_left'},axis=1),fill_value=0)
    annotations_df['bb_left'] = temp1 / 2
    temp2 = annotations_df['bb_top'].add(annotations_df['y_max'].rename({'y_max':'bb_top'},axis=1),fill_value=0)
    annotations_df['bb_top'] = temp2 / 2
    annotations_df = annotations_df.rename({'bb_left':'x_center', 'bb_top':'y_center'}, axis=1)
    return pd.concat([annotations_df['x_center'], annotations_df['y_center']],axis=1)

def track_possession(df):
    num_frames = len(df)
    # num_frames = 100

    team1_poss_frame_count = 0

    for frame in range(num_frames):
        team1_positions = np.array([df.iloc[frame, :11]['x_center'].to_numpy(), df.iloc[frame, 23:34]['y_center'].to_numpy()])
        team2_positions = np.array([df.iloc[frame, 11:22]['x_center'].to_numpy(), df.iloc[frame, 34:45]['y_center'].to_numpy()])
        ball_position = np.array([df.iloc[frame, 22:23]['x_center'], df.iloc[frame, 45:46]['y_center']])

        team1_dist_to_ball = np.absolute(np.array([team1_positions[0] - ball_position[0], team1_positions[1] - ball_position[1]]))
        team2_dist_to_ball = np.absolute(np.array([team2_positions[0] - ball_position[0], team2_positions[1] - ball_position[1]]))

        team1_min_dist = team_min_dist(team1_dist_to_ball)
        team2_min_dist = team_min_dist(team2_dist_to_ball)

        if team1_min_dist < team2_min_dist:
            team1_poss_frame_count += 1

    print(f"Team 1 had possession {100 * team1_poss_frame_count / num_frames : .2f}% of the time ({team1_poss_frame_count / 30 : .2f} out of {num_frames / 30 : .2f} total seconds)")

def team_min_dist(team_dist_to_ball):
    min_dist = -1

    for i in range(len(team_dist_to_ball[0])):
        player_x_dist = team_dist_to_ball[0][i]
        player_y_dist = team_dist_to_ball[1][i]
        new_dist = np.sqrt(player_x_dist**2 + player_y_dist**2)

        if min_dist == -1 or new_dist < min_dist:
            min_dist = new_dist

    return min_dist    

def main():
    annotations_dir = './data/top_view/annotations'
    videos_dir = './data/top_view/videos'
    img_dir = './data/top_view/frames'
    annotations_df = prep(annotations_dir, videos_dir, img_dir, save_frames=False, start_vid=0, end_vid=1)

    centers_df = centers_only(annotations_df)

    track_possession(centers_df)

    make_tracking_gif(centers_df)

if __name__ == "__main__":
    main()