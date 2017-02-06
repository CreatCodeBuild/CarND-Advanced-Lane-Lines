import cv2
import os
import numpy as np


class Calibrator:

	def __init__(self):
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
				print('discard chessboard image', i)  # something went wrong, abort

	def undistort(self, image):
		ret, mtx, dist, rvecs, tvecs = \
			cv2.calibrateCamera(self.object_points, self.image_points, image.shape[:2][::-1], None, None)
		return cv2.undistort(image, mtx, dist, None, mtx)


def s_threshold(img, color_space='BGR'):
	"""
	binary threshold this image
	:param         img: a colored image
	:param color_space: color space of the input image. BGR or RGB
	:return: thresholded image
	"""
	if color_space == 'BGR':
		img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	elif color_space == 'RGB':
		img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	elif color_space == 'HLS':
		pass
	else:
		raise Exception('Color Space Error')

	# get S channel
	img = img[:, :, 2]
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.blur(img, (3, 3))

	# cv2.imshow('', img)
	# cv2.waitKey(10000)

	thresh = (170, 255)
	binary = np.zeros_like(img)
	binary[(img > thresh[0]) & (img <= thresh[1])] = 255
	return binary


def gradient_threshold(img, color_space='BGR'):
	if color_space == 'BGR':
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	elif color_space == 'RGB':
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	else:
		raise Exception('Color Space Error')

	# Sobel x
	sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
	abs_sobel_x = np.absolute(sobel_x)  # Absolute x derivative to accentuate lines away from horizontal
	scaled_sobel = np.uint8(255 * abs_sobel_x / np.max(abs_sobel_x))

	# cv2.imshow('', scaled_sobel)
	# cv2.waitKey(10000)

	# Threshold x gradient
	thresh_min = 20
	thresh_max = 80
	sobel_x_binary = np.zeros_like(scaled_sobel)
	sobel_x_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 255
	return sobel_x_binary


def combined_threshold(img, color_space='BGR'):
	s_binary = s_threshold(img, color_space)
	sober_x_binary = gradient_threshold(img, color_space)
	# Stack each channel to view their individual contributions in green and blue respectively
	# This returns a stack of the two binary images, whose components you can see as different colors
	color_binary = np.dstack((np.zeros_like(sober_x_binary), sober_x_binary, s_binary))
	cv2.imshow('', color_binary)
	cv2.waitKey(10000)

	# Combine the two binary thresholds
	combined_binary = np.zeros_like(sober_x_binary)
	combined_binary[(s_binary == 255) | (sober_x_binary == 255)] = 255
	return combined_binary

# def perspective_transform(img, )



