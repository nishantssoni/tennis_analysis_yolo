from ultralytics import YOLO
import cv2
import torch
import pickle
import sys
sys.path.append("../")
from utils import mesure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def choose_and_filter_players(self, court_keypoints,player_detections):
        # get the first frame
        player_detections_first_frame = player_detections[0]
        # choose the two players whose distance to the center of the courts are the smallest
        chosen_players = self.choose_player(court_keypoints,player_detections_first_frame)
        # filter the other players
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_players}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_player(self, court_keypoints,player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)
            # calculate the distance between the player and the center of the court
            min_distance = float('inf')

            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = mesure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance

            distances.append((track_id, min_distance))

        distances.sort(key=lambda x: x[1])
        # choose the first two persons whose distance to the center of the courts are the smallest
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players
            
    
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections
        
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True, device=self.device)[0]
        id_names = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_names[object_cls_id]
            if object_cls_name == 'person':
                player_dict[track_id] = result

        return player_dict
    
    def draw_bboxes(self, frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(frames, player_detections):
            # draw the bounding box
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, 'person id: ' + str(track_id), (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            output_video_frames.append(frame)

        return output_video_frames