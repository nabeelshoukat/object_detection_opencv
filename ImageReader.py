# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests

frame = cv2.imread('new2.jpeg')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (11, 11), 0)
canny = cv2.Canny(blur, 30, 150, 3)
dilated = cv2.dilate(canny, (1, 1), iterations=0)

(cnt, hierarchy) = cv2.findContours(
    dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
if len(cnt) == 0:
    print("objects in image : ", len(cnt))
else:
    print("objects in image : ", len(cnt) - 1)
while True:
    # cv2.imshow('frame', frame)
    # cv2.imshow('canny', canny)
    # cv2.imshow('dilated', dilated)
    cv2.imshow('RGB', rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Destroy all the windows
cv2.destroyAllWindows()

# Destroy all the windows
# cv2.destroyAllWindows()
