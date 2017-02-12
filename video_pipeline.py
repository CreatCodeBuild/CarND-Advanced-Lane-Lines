from moviepy.editor import VideoFileClip
from helper import *


calibrator = Calibrator()
transformer = Transformer()


def pipeline(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    undistorted = calibrator.undistort(image)
    thresholded = combined_threshold(undistorted)
    warped, perspective_transform_matrix, inverse_perspective_transform_matrix = transformer.transform(thresholded)
    left_fitx, right_fitx, ploty, left_fit, right_fit, leftx, rightx, lefty, righty = find_lines(warped)
    result = project_back(warped, image, undistorted, inverse_perspective_transform_matrix,
                          left_fitx, right_fitx, ploty)
    radius = find_radius(ploty, left_fit, right_fit, leftx, rightx, lefty, righty)
    cv2.putText(result, 'Radius: ' + str(radius) + ' m', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
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
    video_pipeline("project_video.mp4", "project_video_out.mp4", pipeline)
