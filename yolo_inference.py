from ultralytics import YOLO
import torch
print(torch.cuda.is_available())  # Should return True if a compatible GPU is found
print(torch.cuda.device_count())   # Should return the number of GPUs detected


# model = YOLO("yolov8x") 
model = YOLO("models/last.pt")
model.to('cuda')
result = model.predict('input_videos/input_video.mp4', save=True, device='cuda')

print(result)

print('boxex')
for box in result[0].boxes:
    print(box)

    