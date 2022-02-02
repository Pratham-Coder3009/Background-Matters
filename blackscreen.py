import cv2
import numpy as np
import time

from sklearn.datasets import make_hastie_10_2

fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_file = cv2.VideoWriter("output.avi",fourcc,20.0,(640,480))
# Starting the webcam
cap = cv2.VideoCapture(0)
time.sleep(2)
bg = 0
#Capturing the backgroud for 60frames
for i in range(60):
    ret,bg = cap.read()

bg = np.flip(bg, axis=1)

while(cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break
    img = np.flip(img, axis=1)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_black = np.array(0,0,0)
upper_black = np.array(0,0,0)
mask_2 = cv2.inRange(hsv, lower_black, upper_black)

mask_1 = mask_1 + mask_2

mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3,3), np.unit8))
mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3,3), np.unit8))

mask2 = cv2.bitwise_not(mask_1)

res_1 = cv2.bitwise_and(img, img, mask=mask_2)
res_2 = cv2.bitwise_and(bg, bg, mask=mask_1)
final_output = cv2.addWeighted(res_1, 1, res_2, 1, 0)
output_file.write(final_output)
cv2.imshow("magic", final_output)
cv2.waitKey(1)

cap.realese()
cv2.destroyAllWindows()