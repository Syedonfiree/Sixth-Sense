import cv2
import numpy as np
import os

# =========================
# Paths / settings
# =========================
CALIB_PATH = "calibration/stereo_calibration.npz"
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480
FPS = 30

if not os.path.exists(CALIB_PATH):
    print(f"ERROR: Calibration file not found: {CALIB_PATH}")
    exit()

# =========================
# Load calibration
# =========================
data = np.load(CALIB_PATH)

M1 = data["M1"]
d1 = data["d1"]
M2 = data["M2"]
d2 = data["d2"]
R1 = data["R1"]
R2 = data["R2"]
P1 = data["P1"]
P2 = data["P2"]

print("Loaded calibration file successfully.")

# =========================
# Open stereo camera
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Warm up
frame = None
good_reads = 0

for _ in range(20):
    ret, test_frame = cap.read()
    if ret and test_frame is not None:
        frame = test_frame
        good_reads += 1
    cv2.waitKey(10)

if frame is None or good_reads < 5:
    print("ERROR: Camera opened but did not stream reliably.")
    cap.release()
    exit()

h, w = frame.shape[:2]
eye_w = w // 2
eye_h = h

print(f"Actual stereo frame: {w} x {h}")
print(f"Per-eye size: {eye_w} x {eye_h}")

# =========================
# Build rectification maps
# =========================
mapL1, mapL2 = cv2.initUndistortRectifyMap(
    M1, d1, R1, P1, (eye_w, eye_h), cv2.CV_32FC1
)

mapR1, mapR2 = cv2.initUndistortRectifyMap(
    M2, d2, R2, P2, (eye_w, eye_h), cv2.CV_32FC1
)

print("Rectification maps created.")

# =========================
# Window setup
# =========================
cv2.namedWindow("Rectified Stereo", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Rectified Stereo", 1200, 500)

# =========================
# Main loop
# =========================
consecutive_failures = 0
MAX_FAILURES = 10

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        consecutive_failures += 1
        print(f"Failed to read frame ({consecutive_failures}/{MAX_FAILURES})")
        if consecutive_failures >= MAX_FAILURES:
            print("Too many frame failures. Exiting.")
            break
        continue

    consecutive_failures = 0

    left = frame[:, :eye_w]
    right = frame[:, eye_w:]

    rectL = cv2.remap(left, mapL1, mapL2, cv2.INTER_LINEAR)
    rectR = cv2.remap(right, mapR1, mapR2, cv2.INTER_LINEAR)

    combined = cv2.hconcat([rectL, rectR])

    # Draw horizontal epipolar lines
    for y in range(0, eye_h, 40):
        cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)

    # Labels
    cv2.putText(combined, "LEFT RECTIFIED", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(combined, "RIGHT RECTIFIED", (eye_w + 20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Rectified Stereo", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()