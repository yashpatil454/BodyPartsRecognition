import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpHand = mp.solutions.hands
drawSpec = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
cap = cv2.VideoCapture(0)

with mpHand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, img = cap.read()
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img)
        print(results)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS,
                                      drawSpec, drawSpec)

        cv2.imshow('Image', img)
        cv2.waitKey(1)
