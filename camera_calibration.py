import cv2
import os
import numpy as np


class Calibrator:

	def  __init__(self):
		# Arrays to store object points and image points from all the images.
		self.object_points = []		# 3d points in real world space
		self.image_points = []		# 2d points in image plane.

		# Read in images as grayscale
		cameraImages = []
		folder = 'camera_cal'
		for path in os.listdir(folder):
			cameraImages.append(cv2.imread(os.path.join(folder, path), flags=cv2.IMREAD_GRAYSCALE))

		num_col = 9
		num_row = 6
		# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
		objp = np.zeros((6 * 9, 3), np.float32)
		objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
		for i, gray_image in enumerate(cameraImages):
			ret, corners = cv2.findChessboardCorners(gray_image, (num_col, num_row), None)
			if ret:
				self.object_points.append(objp)
				self.image_points.append(corners)
			else:
				print(i)  # something went wrong, abort

	def undistort(self, image):
		ret, mtx, dist, rvecs, tvecs = \
			cv2.calibrateCamera(self.object_points, self.image_points, image.shape[::-1], None, None)
		return cv2.undistort(image, mtx, dist, None, mtx)


calibrator = Calibrator()
undistorted = calibrator.undistort(cv2.imread('test_images/test6.jpg', flags=cv2.IMREAD_GRAYSCALE))

cv2.imshow('123', undistorted)
cv2.waitKey(10000)
