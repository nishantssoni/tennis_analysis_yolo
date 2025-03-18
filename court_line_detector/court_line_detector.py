import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np

class CourtLineDetector:
    def __init__(self, model_path):
        # # set device
        self.model = models.resnet50(pretrained=False)
        # replace the last layer
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        # load weights
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def predict(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = self.transform(frame_rgb).unsqueeze(0)

        with torch.no_grad():
            output = self.model(frame_tensor)
        
        keypoints = output.squeeze().cpu().numpy()
        original_height, original_width = frame.shape[:2]
        keypoints[::2] *= original_width/224.0
        keypoints[1::2] *= original_height/224.0

        return keypoints

    def draw_keypoints(self, image, keypoints):
        # Plot keypoints on the image
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image
    
    def draw_keypoints_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames



























# import torch
# import torchvision.transforms as transforms
# import torchvision.models as models
# import cv2

# class CourtLineDetector:
#     def __init__(self, model_path):
#         # # set device
#         # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         # load model
#         self.model = models.resnet50(pretrained=False)
#         # replace the last layer
#         self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
#         # load weights
#         self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
#         # self.model.to(self.device)

#         self.transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#     # we only predict once because the camera is not moving so we can just predict once
#     def predict(self, frame):
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame = self.transform(frame).unsqueeze(0)

#         with torch.no_grad():
#             output = self.model(frame)
        
#         keypoints = output.squeeze().cpu().numpy()
#         original_height, original_width = frame.shape[:2]
#         keypoints[::2] *= original_width/244.0
#         keypoints[1::2] *= original_height/244.0

#         return keypoints
    
#     def draw_keypoints(self, frame, keypoints):
#         for i in range(0, len(keypoints), 2):
#             x = int(keypoints[i])
#             y = int(keypoints[i+1])
#             cv2.putText(frame, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 2)
#             cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
#         return frame
    
#     def draw_keypoints_video(self, frames, keypoints):
#         output_video_frames = []
#         for frame in frames:
#             frame = self.draw_keypoints(frame, keypoints)
#             output_video_frames.append(frame)
#         return output_video_frames
    