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
	pass

if __name__ == '__main__':
	# test_Calibrator()
	# test_s_threshold()
	# test_gradient_threshold()
	test_combined_threshold()
