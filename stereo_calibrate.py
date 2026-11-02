import cv2
import numpy as np
import glob
import os

# ==============================
# Checkerboard Settings
# ==============================

CHECKERBOARD = (8, 6)   # inner corners
SQUARE_SIZE = 0.025     # meters (25mm example)

LEFT_DIR = "calibration/left/*.png"
RIGHT_DIR = "calibration/right/*.png"

# ==============================
# Prepare object points
# ==============================

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints_left = []
imgpoints_right = []

# ==============================
# Load image pairs
# ==============================

left_images = sorted(glob.glob(LEFT_DIR))
right_images = sorted(glob.glob(RIGHT_DIR))

print(f"Found {len(left_images)} left images")
print(f"Found {len(right_images)} right images")

valid_pairs = 0

for left_path, right_path in zip(left_images, right_images):

    imgL = cv2.imread(left_path)
    imgR = cv2.imread(right_path)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

    if retL and retR:

        objpoints.append(objp)
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)

        valid_pairs += 1

        cv2.drawChessboardCorners(imgL, CHECKERBOARD, cornersL, retL)
        cv2.drawChessboardCorners(imgR, CHECKERBOARD, cornersR, retR)

        display = cv2.hconcat([imgL, imgR])
        cv2.imshow("Detected Checkerboard", display)
        cv2.waitKey(200)

    else:
        print(f"Skipping pair: {left_path}")

cv2.destroyAllWindows()

print("Valid pairs:", valid_pairs)

if valid_pairs < 10:
    print("Not enough valid pairs for calibration")
    exit()

# ==============================
# Calibrate each camera
# ==============================

retL, M1, d1, rvecsL, tvecsL = cv2.calibrateCamera(
    objpoints,
    imgpoints_left,
    grayL.shape[::-1],
    None,
    None
)

retR, M2, d2, rvecsR, tvecsR = cv2.calibrateCamera(
    objpoints,
    imgpoints_right,
    grayR.shape[::-1],
    None,
    None
)

# ==============================
# Stereo Calibration
# ==============================

flags = cv2.CALIB_FIX_INTRINSIC

ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    M1,
    d1,
    M2,
    d2,
    grayL.shape[::-1],
    criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5),
    flags=flags
)

# ==============================
# Stereo Rectification
# ==============================

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    M1,
    d1,
    M2,
    d2,
    grayL.shape[::-1],
    R,
    T
)

# ==============================
# Save calibration
# ==============================

output_path = "calibration/stereo_calibration.npz"

np.savez(
    output_path,
    M1=M1,
    d1=d1,
    M2=M2,
    d2=d2,
    R=R,
    T=T,
    R1=R1,
    R2=R2,
    P1=P1,
    P2=P2,
    Q=Q
)

print("\nCalibration complete")
print("Saved to:", output_path)