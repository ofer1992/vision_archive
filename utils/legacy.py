import cv2
import numpy as np
from utils.etc import imshow
import matplotlib.pyplot as plt

class MotionHistogram:
    """
    Plot and get median of feature points motion
    """

    def __init__(self):
        # self.fig = plt.figure()
        self.c = 0
        self.bins = np.arange(-10,10,0.3)

    def draw(self, p1, p0):
        ' plot histogram of points motion'
        if self.c < 10:
            self.c += 1
            return
        plt.clf()
        self.c = 0
        t = p1 - p0
        t = t.flatten()
        x_mot = t[0::2]
        y_mot = t[1::2]

        plt.hist(x_mot, bins=self.bins, label='x', color='orange', alpha=0.5)
        #         plt.hist(y_mot, bins=self.bins, label='y', color='blue', alpha=0.5)
        # redraw the canvas
        self.fig.canvas.draw()

        # convert canvas to image
        img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8,
                            sep='')
        img  = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        cv2.imshow('hist',img)

    def median(self, p1, p0):
        ' get median of features points motion'
        t = p1 - p0
        t = t.flatten()
        x_mot = t[0::2]
        y_mot = t[1::2]
        # uniq, count = np.unique(x_mot, return_counts=True)
        # mode_x = uniq[np.argmax(count)]
        # uniq, count = np.unique(y_mot, return_counts=True)
        # mode_y = uniq[np.argmax(count)]

        #         return mode_x, mode_y
        if x_mot.size == 0 or y_mot.size == 0:
            return 0, 0
        return np.median(x_mot), np.median(y_mot)


class CornerTracker:

    CORNER_SIZE = 300
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=1,
                          qualityLevel=0.1,
                          minDistance=20,
                          blockSize=7)

    def __init__(self, frame):
        assert len(frame.shape) == 2, "frame must be grayscale"
        self.last_frame = frame
        self.mask = np.zeros(frame.shape[:2], dtype='uint8')
        self.mask[0:self.CORNER_SIZE, :] = 255
        self.point = cv2.goodFeaturesToTrack(frame, mask=self.mask, **self.feature_params)

    def track(self, frame):
        assert len(frame.shape) == 2, "frame must be grayscale"
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.last_frame, frame, self.point, None, **self.lk_params)
        motion = np.squeeze(p1 - self.point)
        if st[0] >= 0:
            # self.point = cv2.goodFeaturesToTrack(frame, mask=self.mask, **self.feature_params)
            self.point = cv2.goodFeaturesToTrack(self.last_frame[0:self.CORNER_SIZE, :], mask=None, **self.feature_params)
        else:
            self.point = p1
        assert self.point is not None, "No points generated!"
        self.last_frame = frame
        return motion

    def get_corner(self):
        window = self.last_frame[0:self.CORNER_SIZE, :].copy()
        cv2.circle(window, tuple(np.squeeze(self.point)), 3, (0, 0, 0), -1)
        return window


class OpticalFlowTracker1:
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=10,
                          qualityLevel=0.6,
                          minDistance=20,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def __init__(self, frame, mask=None):
        """ assume frame is gray """
        assert len(frame.shape) == 2, "frame must be grayscale"
        self.mask = mask
        self.old = frame.copy()
        self.p0 = cv2.goodFeaturesToTrack(frame, mask=mask, **self.feature_params)
        # Create some random colors
        self.color = np.random.randint(0, 255, (100, 3))
        self.motion_hist = MotionHistogram()

    def calc_flow(self, frame):
        assert len(frame.shape) == 2, "frame must be grayscale"
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old, frame, self.p0, None, **self.lk_params)
        self.old = frame.copy()
        good_new = p1[st == 1]
        good_old = self.p0[st == 1]
        mode_x, mode_y = self.motion_hist.median(good_new, good_old)
        self.p0 = good_new.reshape(-1, 1, 2)
        if self.p0.shape[0] < 5:
            mask_points = np.zeros_like(self.mask, dtype='uint8')
            # TODO: prevent redundant feature points. disabled for now
            # y, x = np.indices(mask_points.shape)
            # for p in self.p0:
            #     p = p.squeeze()
            #     print p
            #     mask_points[(np.abs(x - p[0]) < 50) & (np.abs(y - p[1]) < 50)] = 255
            new_points = cv2.goodFeaturesToTrack(frame, mask=cv2.subtract(self.mask, mask_points), **self.feature_params)
            if new_points is not None:
                self.p0 = np.vstack((self.p0, new_points))

        return mode_x, mode_y

    def draw_points_on_frame(self, frame):
        for i, point in enumerate(self.p0):
            cv2.circle(frame, tuple(point.squeeze()), 3, self.color[i], -1)


class TemplateTracker:
    """ Generates template from starting coordinates and tracks in video"""
    TEMPLATE_WIDTH = 50
    TEMPLATE_HIGHT = 50
    BASE_SEARCH_DELTA = 100

    def __init__(self, frame, point):
        assert len(frame.shape) == 2, "frame must be grayscale"
        self.x = point[0]
        self.y = point[1]
        self.delta_inc = 0
        self.TEMPLATE_WIDTH = int(min(self.TEMPLATE_WIDTH, 2*self.x, 2*(frame.shape[1]-self.x)))
        self.TEMPLATE_HIGHT = int(min(self.TEMPLATE_HIGHT, 2*self.y, 2*(frame.shape[1]-self.y)))
        self._generate_template(frame, int(self.x), int(self.y))

    def _generate_template(self, frame, x, y):
        'create template from frame with center at x,y'
        #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dx = self.TEMPLATE_WIDTH / 2
        dy = self.TEMPLATE_HIGHT / 2
        self.template = frame[y - dy:y + dy, x - dx:x + dx]

    def track_new_loc(self, frame):
        assert len(frame.shape) == 2, "frame must be grayscale"
        delta = self.BASE_SEARCH_DELTA + self.delta_inc
        min_row = max(0, int(self.y) - delta)
        max_row = min(int(self.y) + delta, frame.shape[0] - 1)
        min_col = max(0, int(self.x) - delta)
        max_col = min(int(self.x) + delta, frame.shape[1] - 1)
        #         print min_row, max_row, min_col, max_col
        sub_frame = frame[min_row:max_row,
                    min_col:max_col]
        assert sub_frame.shape[0] >= self.template.shape[0], str(sub_frame.shape) + " " + str(self.template.shape)
        assert sub_frame.shape[1] >= self.template.shape[1]
        res = cv2.matchTemplate(sub_frame, self.template, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = list(max_loc)
        if max_val > 0.95:
            #         cv2.circle(sub_frame, tuple(top_left), 10, (0,0,0), -1)
            #         bottom_right = tuple([top_left[0]+self.TEMPLATE_WIDTH, top_left[1]+self.TEMPLATE_HIGHT])
            #         cv2.rectangle(sub_frame, tuple(top_left), bottom_right, 255, 2)
            #         imshow("sub_frame", sub_frame)
            #         imshow("conv", res, gray=True)
            self.x = top_left[0] + min_col + self.TEMPLATE_WIDTH / 2
            self.y = top_left[1] + min_row + self.TEMPLATE_HIGHT / 2
            self.delta_inc = 0
        else:
            self.delta_inc = min(self.delta_inc + 1, min(frame.shape))

    def apply_motion(self, dx, dy):
        self.x += dx
        self.y += dy

    def draw_rect(self, frame):
        top_left = (int(self.x) - self.TEMPLATE_WIDTH / 2, int(self.y) - self.TEMPLATE_HIGHT / 2)
        bottom_right = (int(self.x) + self.TEMPLATE_WIDTH / 2, int(self.y) + self.TEMPLATE_HIGHT / 2)
        cv2.circle(frame, self.get_loc(), 3, (0, 0, 0), -1)
        cv2.rectangle(frame, top_left, bottom_right, (255,0,0), 2)
        delta = self.BASE_SEARCH_DELTA + self.delta_inc
        delta_top_left = (int(self.x) - delta / 2, int(self.y) - delta / 2)
        delta_bottom_right = (int(self.x) + delta / 2, int(self.y) + delta / 2)
        cv2.rectangle(frame, delta_top_left, delta_bottom_right, (0,0,255), 2)

    def show_template(self):
        imshow(self.template, "template")

    def get_loc(self):
        return (int(self.x), int(self.y))



class BoardTracker:
    """
    tracks corners using motion of background (under the false uniform pixel motion
    assumption).
    Further refinement of visible corners is done using the TemplateTracker.
    Generates rectified board including non-visible part which is kept from
    past frames.
    """
    board_id = 0
    RECTIFIED_CORNERS = {'tl':(0,0),
                         'bl':(0,300),
                         'tr':(750,0),
                         'br':(750,300)}

    def __init__(self, frame, tl, tr, bl, br):
        self.board_id = BoardTracker.board_id
        BoardTracker.board_id += 1
        self.corner_trackers = {
            'tl': TemplateTracker(frame, tl),
            'tr': TemplateTracker(frame, tr),
            'bl': TemplateTracker(frame, bl),
            'br': TemplateTracker(frame, br)
        }
        self.bg = None
        self.counter = 0
        self.moving = False

    def apply_motion(self, dx, dy):
        """
        move all corners by specified motion.

        this is the calculated motion of the background, under the
        assumption of uniform motion under rotation.
        """
        for t in self.corner_trackers:
            self.corner_trackers[t].apply_motion(dx, dy)

    def apply_template_tracking(self, frame):
        """
        refine location estimation of corners using template tracking.
        """
        for t in self.corner_trackers.values():
            if 0 < t.x < frame.shape[1] and 0 < t.y < frame.shape[0]:
                t.track_new_loc(frame)

    def draw_corners(self, frame):
        """
        draw visible corners on image
        """
        for t in self.corner_trackers.values():
            if 0 < t.x < frame.shape[1] and 0 < t.y < frame.shape[0]:
                t.draw_rect(frame)

    def rectify(self, frame, camera_moving):
        x0 = frame.shape[1]
        y0 = frame.shape[0]
        big = np.zeros(tuple([i * 3 for i in frame.shape[:2]]+[frame.shape[2]]), dtype='uint8')
        big[y0:y0+frame.shape[0],x0:x0+frame.shape[1]] = frame
        pts1 = np.float32([[x0+self.corner_trackers[c].x, y0+self.corner_trackers[c].y] for c in self.corner_trackers])
        pts2 = np.float32([self.RECTIFIED_CORNERS[c] for c in self.corner_trackers])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(big,M,(750,300))
        if self.bg is None:
            self.bg = dst
        elif self.counter % 25 == 0 and not camera_moving:
            self.bg[dst!=0] = dst[dst!=0]
        dst[dst==0] = self.bg[dst==0]
        self.counter+=1

        cv2.imshow("board #"+str(self.board_id), dst)
        # out.write(dst)

