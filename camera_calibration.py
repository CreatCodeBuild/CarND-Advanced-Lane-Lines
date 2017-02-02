import cv2
import os


# Read In Images
cameraImages = []
folder = 'camera_cal'
for path in os.listdir(folder):
	cameraImages.append(cv2.imread(os.path.join(folder, path), flags=cv2.IMREAD_COLOR))

# # print(cameraImages[0])
# cv2.imshow('123', cameraImages[0])
# cv2.waitKey(1000)
