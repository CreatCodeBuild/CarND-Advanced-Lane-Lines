import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

debug = False


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
	img = cv2.medianBlur(img, 3)

	thresh = (170, 255)
	binary = np.zeros_like(img)
	binary[(img > thresh[0]) & (img <= thresh[1])] = 255
	# cv2.imshow('', binary)
	# cv2.waitKey(10000)
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

	# Threshold x gradient
	thresh_min = 20
	thresh_max = 100
	sobel_x_binary = np.zeros_like(scaled_sobel)
	sobel_x_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 255
	# cv2.imshow('', sobel_x_binary)
	# cv2.waitKey(10000)
	return sobel_x_binary


def combined_threshold(img, color_space='BGR'):
	"""
	combine s_threshold and gradient_threshold
	"""
	s_binary = s_threshold(img, color_space)
	sober_x_binary = gradient_threshold(img, color_space)
	# Stack each channel to view their individual contributions in green and blue respectively
	# This returns a stack of the two binary images, whose components you can see as different colors
	# color_binary = np.dstack((np.zeros_like(sober_x_binary), sober_x_binary, s_binary))
	# cv2.imshow('', color_binary)
	# cv2.waitKey(10000)

	# Combine the two binary thresholds
	combined_binary = np.zeros_like(sober_x_binary)
	combined_binary[(s_binary == 255) | (sober_x_binary == 255)] = 255
	# cv2.imshow('', combined_binary)
	# cv2.waitKey(10000)
	return combined_binary


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


class Transformer:
	def __init__(self):
		img_size = (1280, 720)
		mid = img_size[0]/2
		top = 80
		bottom = 450
		self.points = [
			(mid-top, 470),
			(mid+top, 470),
			(mid+bottom, 710),
			(mid-bottom, 710)
		]
		self.source = np.float32(self.points)
		self.destination = np.float32([
			[mid-bottom, 0],
			[mid+bottom, 0],
			[mid+bottom, 720],
			[mid-bottom, 720]
		])

	def transform(self, img):
		perspective_transform_matrix = cv2.getPerspectiveTransform(self.source, self.destination)
		inverse_perspective_transform_matrix = cv2.getPerspectiveTransform(self.destination, self.source)
		warped = cv2.warpPerspective(img, perspective_transform_matrix, (1280, 720), flags=cv2.INTER_LINEAR)
		return warped, perspective_transform_matrix, inverse_perspective_transform_matrix

	def debug(self):
		a = cv2.imread('output_images/calibration/straight_lines1.jpg')
		print(a.shape)
		a = region_of_interest(a, np.array([self.points]))

		cv2.imshow('', a)
		cv2.waitKey(10000)


def show(a):
	cv2.imshow('', a)
	cv2.waitKey(10000)


def find_lines(binary_warped):
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	# print(binary_warped.shape[0]/2)
	histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0] / 2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0] / nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window + 1) * window_height
		win_y_high = binary_warped.shape[0] - window * window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# Draw the windows on the visualization image
		cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
		cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = (
			(nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &
			(nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = (
			(nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &
			(nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	if debug:
		print('DEBUG')
		# Generate x and y values for plotting
		ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
		left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
		right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

		out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
		plt.imshow(out_img)
		plt.plot(left_fitx, ploty, color='yellow')
		plt.plot(right_fitx, ploty, color='yellow')
		plt.xlim(0, 1280)
		plt.ylim(720, 0)
		plt.show()
	####################################
	return left_fit, right_fit


def search_near_last_frame(binary_warped, left_fit, right_fit):
	# Assume you now have a new warped binary image
	# from the next frame of video (also called "binary_warped")
	# It's now much easier to find line pixels!
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 100
	left_lane_inds = (
		(nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) &
		(nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
	right_lane_inds = (
		(nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
		(nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]
	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
	left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
	right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

	if debug:
		# Create an image to draw on and an image to show the selection window
		out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
		window_img = np.zeros_like(out_img)
		# Color in left and right line pixels
		out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

		# Generate a polygon to illustrate the search window area
		# And recast the x and y points into usable format for cv2.fillPoly()
		left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
		left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
		left_line_pts = np.hstack((left_line_window1, left_line_window2))
		right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
		right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
		right_line_pts = np.hstack((right_line_window1, right_line_window2))

		# Draw the lane onto the warped blank image
		cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
		cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
		result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
		plt.imshow(result)
		plt.plot(left_fitx, ploty, color='yellow')
		plt.plot(right_fitx, ploty, color='yellow')
		plt.xlim(0, 1280)
		plt.ylim(720, 0)
		plt.show()
	return left_fitx, right_fitx, ploty, left_fit, right_fit, leftx, rightx, lefty, righty


def find_radius_and_center(ploty, leftx, rightx, lefty, righty):
	y_eval = np.max(ploty)
	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30 / 720  # meters per pixel in y dimension
	xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
	right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

	# Calculate the new radii of curvature
	left_curve_radius = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
		2 * left_fit_cr[0])
	right_curve_radius = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
		2 * right_fit_cr[0])

	left_y_max = np.argmax(lefty)
	right_y_max = np.argmax(righty)
	center_x = (leftx[left_y_max] + rightx[right_y_max])/2
	center_offset = (640 - center_x) * xm_per_pix
	return np.mean([left_curve_radius, right_curve_radius]), center_offset


def project_back(binary_warped, original_image, undistorted_image, inverse_perspective_transform_matrix,
				 left_fitx, right_fitx, ploty):
	# Create an image to draw the lines on
	warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp,
								  inverse_perspective_transform_matrix,
								  (original_image.shape[1], original_image.shape[0]))
	# Combine the result with the original image
	result = cv2.addWeighted(undistorted_image, 1, newwarp, 0.3, 0)
	# show(result)
	return result





# find_lines('output_images/calibration/straight_lines2.jpg')
# find_lines('output_images/calibration/test1.jpg')
# find_lines('output_images/calibration/test2.jpg')
# find_lines('output_images/calibration/test3.jpg')
# find_lines('output_images/calibration/test4.jpg')
# find_lines('output_images/calibration/test5.jpg')
# find_lines('output_images/calibration/test6.jpg')

