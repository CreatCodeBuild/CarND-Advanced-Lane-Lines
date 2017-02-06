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


def threshold(img, color_space='BGR'):
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
		print('Color Space Error')
		exit()

	# get S channel
	img = img[:, :, 2]
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# img = cv2.blur(img, (7, 7))

	cv2.imshow('', img)
	cv2.waitKey(10000)

	thresh = (240, 255)
	binary = np.zeros_like(img)
	binary[(img > thresh[0]) & (img <= thresh[1])] = 255

	return binary


# def perspective_transform(img, )


def test_Calibrator():
	calibrator = Calibrator()
	folder = 'camera_cal'
	for path in os.listdir(folder):
		image = cv2.imread(os.path.join(folder, path), flags=cv2.IMREAD_COLOR)
		undistorted = calibrator.undistort(image)
		cv2.imwrite(os.path.join('output_images/calibration', path), undistorted)
	folder = 'test_images'
	for path in os.listdir(folder):
		image = cv2.imread(os.path.join(folder, path), flags=cv2.IMREAD_COLOR)
		undistorted = calibrator.undistort(image)
		cv2.imwrite(os.path.join('output_images/calibration', path), undistorted)
	print('Calibrator test is done')


def test_threshold():
	img = threshold(cv2.imread('test_images/test6.jpg', cv2.IMREAD_COLOR))
	cv2.imshow('', img)
	cv2.waitKey(10000)

def test_perspective_transform():
	pass
if __name__ == '__main__':
	# test_Calibrator()
	test_threshold()
