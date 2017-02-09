import cv2
import os

import numpy
import numpy as np
from scipy.misc import imresize


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
	binary threshold this image using saturation channel(s channel)
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
	"""
	use Sobel operator to take derivative in x(horizontal) direction
	then threshold the gradient image
	:param         img: a colored image
	:param color_space: color space of the input image. BGR or RGB
	:return: thresholded image
	"""
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
	"""
	combine s_threshold and gradient_threshold
	"""
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


def perspective_transform(img):
	# first I need to find 4 points
	pass
def region_of_interest(img, vertices):
	"""
	Applies an image mask.

	Only keeps the region of the image defined by the polygon
	formed from `vertices`. The rest of the image is set to black.
	"""
	# defining a blank mask to start with
	mask = np.zeros_like(img)

	# defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if len(img.shape) > 2:
		channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255

	# filling pixels inside the polygon defined by "vertices" with the fill color
	cv2.fillPoly(mask, vertices, ignore_mask_color)

	# returning the image only where mask pixels are nonzero
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image


class Transformer():
	def __init__(self):
		mid = 640
		top = 80
		bottom = 450
		self.points = [
			(mid-top, 470),
			(mid+top, 470),
			(mid+bottom, 710),
			(mid-bottom, 710)
		]
		self.source = np.float32(self.points)

		img_size = (1280, 720)
		self.destination = np.float32([
			[mid-bottom, 0],
			[mid+bottom, 0],
			[mid+bottom, 720],
			[mid-bottom, 720]
		])

	def transform(self, img):
		M = cv2.getPerspectiveTransform(self.source, self.destination)
		warped = cv2.warpPerspective(img, M, (1280, 720), flags=cv2.INTER_LINEAR)
		cv2.imshow('', warped)
		cv2.waitKey(10000)

	def test(self):
		a = cv2.imread('output_images/calibration/straight_lines1.jpg')
		print(a.shape)
		a = region_of_interest(a, numpy.array([self.points]))

		cv2.imshow('', a)
		cv2.waitKey(10000)

t = Transformer()
a = cv2.imread('output_images/calibration/straight_lines1.jpg')
t.transform(a)
a = cv2.imread('output_images/calibration/straight_lines2.jpg')
t.transform(a)
a = cv2.imread('output_images/calibration/test1.jpg')
t.transform(a)
a = cv2.imread('output_images/calibration/test2.jpg')
t.transform(a)
a = cv2.imread('output_images/calibration/test3.jpg')
t.transform(a)
a = cv2.imread('output_images/calibration/test4.jpg')
t.transform(a)
a = cv2.imread('output_images/calibration/test5.jpg')
t.transform(a)
a = cv2.imread('output_images/calibration/test6.jpg')
t.transform(a)
t.test()