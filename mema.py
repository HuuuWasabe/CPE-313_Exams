import cv2
import numpy as np
import torch
import time
import threading
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

sequence_length = SEQUENCE_LENGTH  
resize_shape = (SIZE, SIZE)    
file_dir = './test_case/OverheadPress.mp4'    
device = "cuda" if torch.cuda.is_available() else "cpu"

cap = cv2.VideoCapture(file_dir)
frames = []
predicted_label = ""

def run_inference(frames_batch):
    global predicted_label
    frames_array = np.stack(frames_batch, axis=0) / 255.0
    frames_tensor = torch.from_numpy(frames_array).float().permute(0, 3, 1, 2).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = load_model(frames_tensor)
    
    predicted_index = torch.argmax(prediction, dim=1).item()
    predicted_label = sorted(LABELS)[predicted_index]
    print("Activity:", predicted_label)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Resize frame before pose detection
    small_frame = cv2.resize(frame, (256, 256))
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Draw landmarks less frequently
    if results.pose_landmarks and cap.get(cv2.CAP_PROP_POS_FRAMES) % 5 == 0:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Prepare frame for model input
    resized_frame = cv2.resize(frame, resize_shape)
    rgb_resized_frame = resized_frame[:, :, ::-1]
    frames.append(rgb_resized_frame)

    # Display predicted activity
    display_text = f'Activity: {predicted_label}'
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Cam', frame)

    # Run inference every SEQUENCE_LENGTH frames
    if len(frames) == SEQUENCE_LENGTH:
        threading.Thread(target=run_inference, args=(frames.copy(),)).start()
        frames = []

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    time.sleep(0.005)  # Limit CPU usage

cap.release()
cv2.destroyAllWindows()
