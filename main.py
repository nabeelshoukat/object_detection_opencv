# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    canny = cv2.Canny(blur, 30, 150, 3)
    dilated = cv2.dilate(canny, (1, 1), iterations=0)

    (cnt, hierarchy) = cv2.findContours(
        dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
    cv2.putText(rgb, f'Total Objects in Frame: {len(cnt)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
    # if len(cnt) == 0:
    #     print("objects in image : ", len(cnt))
    # else:
    #     print("objects in image : ", len(cnt) - 1)
    #
    cv2.imshow('frame', frame)
    # cv2.imshow('canny', canny)
    # cv2.imshow('dilated', dilated)
    cv2.imshow('RGB', rgb)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
