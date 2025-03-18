from ultralytics import YOLO
import cv2
import torch
import pickle
import pandas as pd
import sys
import numpy as np
sys.path.append("../")
from utils import get_center_of_bbox

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def interpolate_ball_positions(self, ball_detections):
        # when there is no detection the list is empty
        ball_positions = [x.get(1,[]) for x in ball_detections]

        # convert the list of lists to a dataframe
        ball_positions_df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # interpolate the missing values
        ball_positions_df = ball_positions_df.interpolate()
        # fill the first row, because interpolation does not fill the first row
        ball_positions_df = ball_positions_df.bfill()

        # convert the dataframe back to a list of lists
        ball_positions = ball_positions_df.values.tolist()
        # convert the list back to the dictionary
        ball_positions = [{1: x} for x in ball_positions]

        return ball_positions
    
    
    def draw_ball_paths(self, frames, ball_positions):
        # create a canvas as the same size as the first frame
        path_canvas = frames[0].copy() * 0 # create a black canvas

        output_video_frames = []
        for i in range(1, len(ball_positions)):
            if ball_positions[i-1] is not None and ball_positions[i] is not None:
                prev_frame_ball = get_center_of_bbox(ball_positions[i-1][1])
                curr_frame_ball = get_center_of_bbox(ball_positions[i][1])
                cv2.line(path_canvas, prev_frame_ball, curr_frame_ball, (0, 0, 255), 2)
            
            # combined the current frame with the path canvas
            combined_frame = cv2.add(frames[i], path_canvas)

            output_video_frames.append(combined_frame)
        return output_video_frames
    
    def draw_velocity_direction(self, frames, ball_positions):
        """ it draws the velocity and direction of the ball speed green and slow red"""
        for i in range(1, len(ball_positions)):
            if ball_positions[i-1] is not None and ball_positions[i] is not None:
                prev_pos = get_center_of_bbox(ball_positions[i-1][1])
                curr_pos = get_center_of_bbox(ball_positions[i][1])
                
                # Calculate velocity and direction
                velocity = int(((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2) ** 0.5)
                color = (0, min(255, velocity * 10), 255 - min(255, velocity * 10))
                
                # Draw arrow for direction
                cv2.arrowedLine(frames[i], prev_pos, curr_pos, color, 2)
        return frames

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections
        
        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame,conf=0.15, device=self.device)[0]
 
        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result

        return ball_dict
    
    def draw_bboxes(self, frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(frames, ball_detections):
            # draw the bounding box
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, 'ball id: ' + str(track_id), (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,255), 2)

            output_video_frames.append(frame)

        return output_video_frames
    
    def get_ball_hit_frames(self, ball_detections):
        # when there is no detection the list is empty
        ball_positions = [x.get(1,[]) for x in ball_detections]

        # convert the list of lists to a dataframe
        ball_positions_df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])        # when there is no detection the list is empty

        # finding middle value of bounding box
        ball_positions_df['mid_y'] = ball_positions_df['y1'] + (ball_positions_df['y2'] - ball_positions_df['y1']) / 2
        
        # mid y rolling mean
        ball_positions_df['mid_y_rolling'] = ball_positions_df['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        # calculate delta y
        ball_positions_df['delta_y'] = ball_positions_df['mid_y_rolling'].diff()

        ball_positions_df['hit'] = 0
        minimum_change_for_hit = 25

        # detect hits
        for i in range(1, len(ball_positions_df)-int(minimum_change_for_hit*1.2)):
            negative_position_change = ball_positions_df['delta_y'].iloc[i] > 0 and ball_positions_df['delta_y'].iloc[i+1]<0
            positive_position_change = ball_positions_df['delta_y'].iloc[i] < 0 and ball_positions_df['delta_y'].iloc[i+1]>0

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i+1, i+int(minimum_change_for_hit*1.2)):
                    negative_position_change_following_frame = ball_positions_df['delta_y'].iloc[i] > 0 and ball_positions_df['delta_y'].iloc[change_frame]<0
                    positive_position_change_following_frame = ball_positions_df['delta_y'].iloc[i] < 0 and ball_positions_df['delta_y'].iloc[change_frame]>0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count +=1
                    if positive_position_change and positive_position_change_following_frame:
                        change_count +=1

                    if change_count > minimum_change_for_hit - 1:
                        # ball_positions_df['hit'].iloc[i] = 1
                        ball_positions_df.loc[i, "hit"] = 1

        frame_numbers_with_ball_hit = ball_positions_df[ball_positions_df['hit'] == 1].index.to_list()

        return frame_numbers_with_ball_hit
    
    def draw_ball_hit_frames(self, frames, frame_numbers_with_ball_hit):
        output_video_frames = []
        for i, frame in enumerate(frames):
            if i in frame_numbers_with_ball_hit:
                cv2.putText(frame, 'ball hit', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames