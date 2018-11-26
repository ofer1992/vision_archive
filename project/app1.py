"""
User must give initial location of board in frame.
Board is tracked using optical flow for rotation tracking,
template tracking for refinement,
Then it is rectified.
"""
import cv2
import numpy as np

from utils.boards import build_board
# from utils.trackers import OptFlowTracker
from utils.legacy import BoardTracker, OpticalFlowTracker1

cap = cv2.VideoCapture("../res/lecture.mp4")

# fast forward to frame with full board in frame
for _ in range(110):
    ret = cap.grab()

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# mask allows choice of feature points only from upper part of frame
corner_mask = np.zeros(old_frame.shape[:2], dtype='uint8')
corner_mask[:75,:] = 255

motion_tracker = OpticalFlowTracker1(old_gray, corner_mask)

# initial corner locations.
corners1 = np.array([[285, 70], [300, 420], [1170, 50], [1160, 400]], dtype='float64')
board1 = BoardTracker(old_gray, tl=corners1[0], bl=corners1[1], tr=corners1[2],
                      br=corners1[3])
boards = [board1]
mask = np.zeros_like(old_frame)

while(1):
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (0,0), fx=0.5,fy=0.5)
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # optical flow
    dx, dy = motion_tracker.calc_flow(frame_gray)
    camera_moving = abs(dx) > 3

    for board in boards:
        board.apply_motion(dx, dy)
        board.apply_template_tracking(frame_gray)
        board.rectify(frame, camera_moving)

    mask = np.uint8(0.9 * mask)
    motion_tracker.draw_points_on_frame(mask)
    for board in boards:
        board.draw_corners(frame)
    cv2.imshow('frame', cv2.add(frame, mask))
    k = cv2.waitKey(1) & 0xff
    if k == ord('b'):
        tmp = build_board(frame)
        if tmp is not None:
            boards.append(BoardTracker(frame_gray, tl=tmp[0], bl=tmp[1], tr=tmp[2],
                          br=tmp[3]))
    if k == ord('q'):
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()

cv2.destroyAllWindows()
cap.release()
