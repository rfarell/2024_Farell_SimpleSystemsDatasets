import cv2
import numpy as np

def video_to_numpy(video_path, output_path, new_height=128, new_width=160):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error opening video file")

    # Read frames from the video
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to grayscale if it's not already
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize the frame
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        frames.append(resized_frame)

    # Convert list of frames to a numpy array
    frames_array = np.array(frames)
    
    # Save the frames as a numpy file
    np.save(output_path, frames_array)

    # Release the video capture object
    cap.release()

# Example usage
video_path = 'pendulum_black_white.mp4'
output_path = 'train.npy'
video_to_numpy(video_path, output_path)
