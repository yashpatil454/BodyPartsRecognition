import cv2
import mediapipe as mp

# import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
drawSpec = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
cap = cv2.VideoCapture(0)

with mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as poses:
    while cap.isOpened():
        success, img = cap.read()
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = poses.process(img)
        # print(results)
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS,
                              drawSpec, mpDraw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
        cv2.imshow('Image', img)
        cv2.waitKey(1)
