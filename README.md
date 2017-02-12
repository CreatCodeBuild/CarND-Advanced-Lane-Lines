## Advanced Lane Finding Report

In this project, I build a pipeline which can detect current lane of a car and compute the lane's area, curvature. The pipeline is robust enough such that it can perform well in various color and lighting conditions of roads and lane lines

My pipeline is composed by the following steps
1. Camera Calibration and Original Camera Image Undistortion
2. Mixed Technique Binary Thresholding of Undistorted Image
3. Bird View Perjection from Forward Facing Image
4. Line Search and Second Order Curve Fitting
5. Curvature Computation
6. Project Results Back to Original Camera Image
