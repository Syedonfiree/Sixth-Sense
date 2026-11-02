#!/usr/bin/env python3
import cv2
import os
import time

# =========================
# Fixed camera settings
# =========================
CAMERA_DEVICE = "/dev/video0"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480
FPS = 30

LEFT_DIR = "calibration/left"
RIGHT_DIR = "calibration/right"

os.makedirs(LEFT_DIR, exist_ok=True)
os.makedirs(RIGHT_DIR, exist_ok=True)

# =========================
# Open camera
# =========================
cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)

if not cap.isOpened():
    print(f"ERROR: Could not open {CAMERA_DEVICE}")
    exit()

# Force known-good mode
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Warm up
print("Warming up camera...")
good_reads = 0
frame = None

for i in range(20):
    ret, test_frame = cap.read()
    if ret and test_frame is not None:
        good_reads += 1
        frame = test_frame
    time.sleep(0.05)

if good_reads < 5 or frame is None:
    print(f"ERROR: Camera opened but unstable ({good_reads}/20 good warmup frames)")
    cap.release()
    exit()

h, w = frame.shape[:2]
print(f"Camera opened successfully: {CAMERA_DEVICE}")
print(f"Actual frame size: {w}x{h}")

if w % 2 != 0:
    print("ERROR: Frame width is not even, cannot split stereo image")
    cap.release()
    exit()

# Smaller, resizable window
cv2.namedWindow("Stereo Capture", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Stereo Capture", 1200, 450)

print("Press SPACE to capture checkerboard pair")
print("Press ESC to quit")

count = 0
consecutive_failures = 0
MAX_FAILURES = 10

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        consecutive_failures += 1
        print(f"Failed to read frame ({consecutive_failures}/{MAX_FAILURES})")
        time.sleep(0.05)

        if consecutive_failures >= MAX_FAILURES:
            print("Too many consecutive frame failures. Exiting.")
            break
        continue

    consecutive_failures = 0

    h, w = frame.shape[:2]
    left = frame[:, :w // 2]
    right = frame[:, w // 2:]

    left_display = left.copy()
    right_display = right.copy()

    cv2.putText(left_display, "LEFT", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(right_display, "RIGHT", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    display = cv2.hconcat([left_display, right_display])
    cv2.imshow("Stereo Capture", display)

    key = cv2.waitKey(1) & 0xFF

    if key == 32:  # SPACE
        left_path = f"{LEFT_DIR}/left_{count:03d}.png"
        right_path = f"{RIGHT_DIR}/right_{count:03d}.png"

        cv2.imwrite(left_path, left)
        cv2.imwrite(right_path, right)

        print(f"Saved pair {count:03d}")
        count += 1

    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()