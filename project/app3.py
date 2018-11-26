"""
using hackyFindBoard
iterate over images in snaps/, rectify boards
take snapshot when information isn't contained (masking black areas)
"""

import numpy as np
import cv2
import os
import utils.boards as boards

def grid(board):
    ' returns grid of "activated" parts in board '
    GRID_SIZE = 75
    ACTIVATION_THRESH = 200
    masks = []
    activation_im = np.zeros((300,750))
    for i in range(4):
        row = []
        for j in range(10):
            mask = np.zeros((300,750))
            mask[i*GRID_SIZE:(i+1)*GRID_SIZE, j*GRID_SIZE: (j+1)*GRID_SIZE] = 1
            if (np.sum(mask*board)/255 > ACTIVATION_THRESH):
                activation_im += mask
            row.append(mask)
        masks.append(row)
    return activation_im


snaps_dir = os.path.abspath("../res/snaps/")+"/"
rectified = {'left': np.zeros((300, 750, 3), dtype='uint8'),
             'right': np.zeros((300,750,3), dtype='uint8')
             }
last = {'left': None, 'right':None}
rectified_points = np.array([[0,0],
                             [0,300],
                             [750, 300],
                             [750, 0]], dtype='float32').reshape(4,1,2)
files = sorted(os.listdir(snaps_dir), key=lambda x: int(x[:x.rindex('.')]))
for frame in (cv2.imread(snaps_dir+f) for f in files):
    cv2.imshow("vid", frame)
    for side, coords in boards.HackyBoardFinder.hackyFindBoard(frame).iteritems():
        print side, coords
        if coords is not None:
            H = cv2.getPerspectiveTransform(coords.reshape(4,1,2),
                                            rectified_points)
            warped = cv2.warpPerspective(frame, H, (750, 300))
            non_black_idx = ~(warped == [0,0,0]).all(axis=2)
            rectified[side][non_black_idx] = warped[non_black_idx]
    for side, rect_board in rectified.iteritems():
        test = cv2.cvtColor(rect_board, cv2.COLOR_BGR2GRAY)
        _, test = cv2.threshold(test, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3),np.uint8)
        test = cv2.dilate(test, kernel, iterations=1)
        if last[side] is not None:
            print np.sum(cv2.subtract(last[side], test))
        last[side] = test
        cv2.imshow(side, test)
        cv2.imshow(side+" act", grid(test))
    k = cv2.waitKey(100) & 0xff
    if k == ord('q'):
        break

cv2.destroyAllWindows()
