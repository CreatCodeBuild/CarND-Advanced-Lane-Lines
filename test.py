"""
test for Class and function in helper.py
"""
from helper import *


def fetch_image(folder):
	"""
	A generator that yield all images under folder as ndarray and their file names
	"""
	for name in os.listdir(folder):
		yield cv2.imread(os.path.join(folder, name), flags=cv2.IMREAD_COLOR), name


def test_Calibrator():
	calibrator = Calibrator()
	for image, file_name in fetch_image('camera_cal'):
		undistorted = calibrator.undistort(image)
		cv2.imwrite(os.path.join('output_images/calibration', file_name), undistorted)
	for image, file_name in fetch_image('test_images'):
		undistorted = calibrator.undistort(image)
		cv2.imwrite(os.path.join('output_images/calibration', file_name), undistorted)
	print('Calibrator test is done')


def test_s_threshold():
	for image, file_name in fetch_image('test_images'):
		image = s_threshold(image)
		cv2.imwrite(os.path.join('output_images/threshold', 's_threshold_'+file_name), image)


def test_gradient_threshold():
	for image, file_name in fetch_image('test_images'):
		image = gradient_threshold(image)
		cv2.imwrite(os.path.join('output_images/threshold', 'gradient_threshold_'+file_name), image)


def test_combined_threshold():
	for image, file_name in fetch_image('test_images'):
		image = combined_threshold(image)
		cv2.imwrite(os.path.join('output_images/threshold', 'combined_threshold_'+file_name), image)


def test_perspective_transform():
	t = Transformer()

	a = cv2.imread('output_images/calibration/straight_lines1.jpg')
	a, _, _ = t.transform(a)
	cv2.imwrite('output_images/transform/straight_lines1.jpg', a)

	a = cv2.imread('output_images/calibration/straight_lines2.jpg')
	a, _, _ = t.transform(a)
	cv2.imwrite('output_images/transform/straight_lines2.jpg', a)

	a = cv2.imread('output_images/calibration/test1.jpg')
	a, _, _ = t.transform(a)
	cv2.imwrite('output_images/transform/test1.jpg', a)

	a = cv2.imread('output_images/calibration/test2.jpg')
	a, _, _ = t.transform(a)
	cv2.imwrite('output_images/transform/test2.jpg', a)

	a = cv2.imread('output_images/calibration/test3.jpg')
	a, _, _ = t.transform(a)
	cv2.imwrite('output_images/transform/test3.jpg', a)

	a = cv2.imread('output_images/calibration/test4.jpg')
	a, _, _ = t.transform(a)
	cv2.imwrite('output_images/transform/test4.jpg', a)

	a = cv2.imread('output_images/calibration/test5.jpg')
	a, _, _ = t.transform(a)
	cv2.imwrite('output_images/transform/test5.jpg', a)

	a = cv2.imread('output_images/calibration/test6.jpg')
	a, _, _ = t.transform(a)
	cv2.imwrite('output_images/transform/test6.jpg', a)

	###
	a = cv2.imread('output_images/threshold/combined_threshold_straight_lines1.jpg')
	a, _, _ = t.transform(a)
	cv2.imwrite('output_images/transform/combined_threshold_straight_lines1.jpg', a)

	a = cv2.imread('output_images/threshold/combined_threshold_straight_lines2.jpg')
	a, _, _ = t.transform(a)
	cv2.imwrite('output_images/transform/combined_threshold_straight_lines2.jpg', a)

	a = cv2.imread('output_images/threshold/combined_threshold_test1.jpg')
	a, _, _ = t.transform(a)
	cv2.imwrite('output_images/transform/combined_threshold_test1.jpg', a)

	a = cv2.imread('output_images/threshold/combined_threshold_test2.jpg')
	a, _, _ = t.transform(a)
	cv2.imwrite('output_images/transform/combined_threshold_test2.jpg', a)

	a = cv2.imread('output_images/threshold/combined_threshold_test3.jpg')
	a, _, _ = t.transform(a)
	cv2.imwrite('output_images/transform/combined_threshold_test3.jpg', a)

	a = cv2.imread('output_images/threshold/combined_threshold_test4.jpg')
	a, _, _ = t.transform(a)
	cv2.imwrite('output_images/transform/combined_threshold_test4.jpg', a)

	a = cv2.imread('output_images/threshold/combined_threshold_test5.jpg')
	a, _, _ = t.transform(a)
	cv2.imwrite('output_images/transform/combined_threshold_test5.jpg', a)

	a = cv2.imread('output_images/threshold/combined_threshold_test6.jpg')
	a, _, _ = t.transform(a)
	cv2.imwrite('output_images/transform/combined_threshold_test6.jpg', a)


def test_find_radius_and_center():
	t = Transformer()

	def inner(name):
		undistorted = cv2.imread('output_images/calibration/' + name + '.jpg')
		thresholded_image = combined_threshold(undistorted, 'BGR')

		warped, perspective_transform_matrix, inverse_perspective_transform_matrix = t.transform(thresholded_image)
		left_fit, right_fit = find_lines(warped)

		left_fitx, right_fitx, ploty, left_fit, right_fit, leftx, rightx, lefty, righty = \
			search_near_last_frame(warped, left_fit, right_fit)

		radius, center_offset = find_radius_and_center(ploty, leftx, rightx, lefty, righty)
		print(radius, center_offset)

	inner('straight_lines1')
	inner('straight_lines2')
	inner('test1')
	inner('test2')
	inner('test3')
	inner('test4')
	inner('test5')
	inner('test6')


def test_project_back():
	t = Transformer()

	def inner(name):
		original = cv2.imread('test_images/'+name+'.jpg')
		undistorted = cv2.imread('output_images/calibration/'+name+'.jpg')
		thresholded_image = combined_threshold(undistorted, 'BGR')
		warped, perspective_transform_matrix, inverse_perspective_transform_matrix = t.transform(thresholded_image)

		left_fit, right_fit = find_lines(warped)
		left_fitx, right_fitx, ploty, left_fit, right_fit, leftx, rightx, lefty, righty = \
			search_near_last_frame(warped, left_fit, right_fit)

		result = project_back(warped, original, undistorted, inverse_perspective_transform_matrix,
							  left_fitx, right_fitx, ploty)

		radius, center_offset = find_radius_and_center(ploty, leftx, rightx, lefty, righty)

		cv2.putText(result, 'Radius: '+str(radius)+' m', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
					bottomLeftOrigin=False)
		cv2.putText(result, 'Center Offset: ' + str(center_offset) + ' m', (50, 100),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
					bottomLeftOrigin=False)
		cv2.imwrite('output_images/project_back/'+name+'.jpg', result)

	inner('straight_lines1')
	inner('straight_lines2')
	inner('test1')
	inner('test2')
	inner('test3')
	inner('test4')
	inner('test5')
	inner('test6')


if __name__ == '__main__':
	# test_Calibrator()
	# test_s_threshold()
	# test_gradient_threshold()
	# test_combined_threshold()
	# test_perspective_transform()
	# test_find_radius_and_center()
	test_project_back()
