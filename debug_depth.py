import numpy as np
import cv2
import os

CALIB_PATH = "calibration/stereo_calibration.npz"
data = np.load(CALIB_PATH)
Q = data["Q"]

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

for _ in range(30):
    cap.read()

ret, frame = cap.read()
cap.release()

eye_w = frame.shape[1] // 2
left  = frame[:, :eye_w]
right = frame[:, eye_w:]

stereo = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=96, blockSize=5,
    P1=600, P2=2400, disp12MaxDiff=1,
    uniquenessRatio=10, speckleWindowSize=100, speckleRange=32
)

grayL     = cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY)
grayR     = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
points_3d = cv2.reprojectImageTo3D(disparity, Q)

Z = points_3d[..., 2]
Y = points_3d[..., 1]
X = points_3d[..., 0]

valid = np.isfinite(Z) & (disparity > 0)
Z_v = Z[valid]
Y_v = Y[valid]

print(f"Z min: {Z_v.min():.3f}m")
print(f"Z 1st percentile: {np.percentile(Z_v, 1):.3f}m")
print(f"Z 5th percentile: {np.percentile(Z_v, 5):.3f}m")
print(f"Z median: {np.median(Z_v):.3f}m")
print(f"Y min: {Y_v.min():.3f}  Y max: {Y_v.max():.3f}")
print(f"Points with Z < 0.3m: {(Z_v < 0.3).sum()}")
print(f"Total valid points: {valid.sum()}")