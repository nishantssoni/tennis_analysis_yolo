import cv2

def read_video(video_path):
    video = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = video.read()
        if ret:
            frames.append(frame)
        else:
            break
    video.release()
    return frames

# save videos

def save_video(frames, video_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(video_path, fourcc, 24, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        video.write(frame)
    video.release()