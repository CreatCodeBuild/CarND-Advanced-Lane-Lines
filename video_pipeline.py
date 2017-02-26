from moviepy.editor import VideoFileClip
from helper import *


calibrator = Calibrator()
transformer = Transformer()

first = True
left_fitx = None
right_fitx = None
ploty = None
left_fit = None
right_fit = None
leftx = None
rightx = None
lefty = None
righty = None

count = 0
recent5_radius = []
recent5_center_offset = []
recent5_left_fitx = []
recent5_right_fitx = []
recent5_ploty = []


def recent5(radius, center_offset, left, right, plot):
    """
    :param radius:
    :param center_offset:
    :param left: left_fitx
    :param right: right_fitx
    :param plot: ploty
    """
    global count
    if count == 5:
        recent5_radius.pop(0)
        recent5_center_offset.pop(0)
        recent5_left_fitx.pop(0)
        recent5_right_fitx.pop(0)
        recent5_ploty.pop(0)
    else:
        count += 1
    recent5_radius.append(radius)
    recent5_center_offset.append(center_offset)
    recent5_left_fitx.append(left)
    recent5_right_fitx.append(right)
    recent5_ploty.append(plot)
    if count > 5:
        print('WEqwdqefd')
        exit()


def pipeline(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    undistorted = calibrator.undistort(image)
    thresholded = combined_threshold(undistorted)
    warped, perspective_transform_matrix, inverse_perspective_transform_matrix = transformer.transform(thresholded)

    global first, left_fitx, right_fitx, ploty, left_fit, right_fit, leftx, rightx, lefty, righty
    if first:
        left_fit, right_fit = find_lines(warped)
        left_fitx, right_fitx, ploty, left_fit, right_fit, leftx, rightx, lefty, righty = \
            search_near_last_frame(warped, left_fit, right_fit)
        first = False

    left_fitx, right_fitx, ploty, left_fit, right_fit, leftx, rightx, lefty, righty = \
        search_near_last_frame(warped, left_fit, right_fit)

    radius, center_offset = find_radius_and_center(ploty, leftx, rightx, lefty, righty)

    recent5(radius, center_offset, left_fitx, right_fitx, ploty)

    result = project_back(warped, image, undistorted, inverse_perspective_transform_matrix,
                          np.add.reduce(recent5_left_fitx) / count,
                          np.add.reduce(recent5_right_fitx) / count,
                          np.add.reduce(recent5_ploty) / count)

    cv2.putText(result, 'Radius: ' + str(np.add.reduce(recent5_radius) / count) + ' m',
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                bottomLeftOrigin=False)
    cv2.putText(result, 'Center Offset: ' + str(np.add.reduce(recent5_center_offset) / count) + ' m', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                bottomLeftOrigin=False)
    
    return result


def video_pipeline(video_file, output_file, process_method):
    """
    :param     video_file: source video file path
    :param    output_file: output video file path
    :param process_method: the callback function which is used to process the video per frame
    :return:
    """
    clip = VideoFileClip(video_file)
    processed_clip = clip.fl_image(process_method)  # NOTE: this function expects color images!!
    processed_clip.write_videofile(output_file, audio=False)


if __name__ == "__main__":
    video_pipeline("project_video.mp4", "project_video_out_new2.mp4", pipeline)
