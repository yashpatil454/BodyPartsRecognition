import time
import cv2

pTime = 0
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
