import numpy as np
import cv2
import time
import tensorflow as tf


# TODO: credit
class DetectorAPI:
    threshold = 0.01
    delta = 50
    def __init__(self, path_to_ckpt='/home/tomer/git/vision_project/NN/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        self.last_mask = None

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        # print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                             int(boxes[0,i,1]*im_width),
                             int(boxes[0,i,2] * im_height),
                             int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def getHumanMask(self, img, first_frame_after_cam_movement=False):
        mask = np.zeros(img.shape[:2], dtype='uint8')
        cv2.resize(img, (1280, 720))
        boxes, scores, classes, num = self.processFrame(img)
        detected_ped = False
        for box, score, cls in zip(boxes, scores, classes):
            # Class 1 represents human
            if cls == 1 and score > self.threshold:
                detected_ped = True
                box = list(box)
                box[0] -= self.delta
                box[1] -= self.delta
                box[2] += img.shape[1]
                box[3] += self.delta
                cv2.rectangle(mask,(box[1],box[0]),(box[3],box[2]),255,-1)
        if not detected_ped and self.last_mask is not None \
                and not first_frame_after_cam_movement:
            # TODO: instead of returning last mask, enlarge rect to account
            # for uncertainty in ped location
            return self.last_mask
        self.last_mask = mask
        return mask

    def close(self):
        self.sess.close()
        self.default_graph.close()


class FeaturePoint:
    ID = 0
    TEMPLATE_WIDTH = 50
    TEMPLATE_HIGHT = 50

    def __init__(self, x, y, frame=None, corner=False):
        """
        feature point in video
        """
        self.ID = FeaturePoint.ID
        FeaturePoint.ID = (FeaturePoint.ID+1) % 100
        self.x = x
        self.y = y
        self.corner = corner
        if frame is not None:
            self.rows, self.cols = frame.shape[:2]
            self.generate_template(frame)

    def tuple(self, integer=False):
        """
        return (x, y) tuple representation of point
        """
        if integer:
            return int(self.x), int(self.y)
        return self.x, self.y

    def score(self, frame):
        """
        get template match score in location
        """
        assert len(frame.shape) == 2, "frame must be grayscale"
        if not self.template_in_frame():
            return -999
        x, y = int(self.x), int(self.y)
        dx, dy = self.TEMPLATE_WIDTH / 2, self.TEMPLATE_HIGHT / 2
        compared_rect = frame[y - dy:y + dy, x - dx:x + dx]
        score = cv2.matchTemplate(compared_rect, self.template, cv2.TM_SQDIFF_NORMED)
        # cv2.imshow(str(self.ID), self.template)
        # cv2.imshow(str(self.ID)+"comp", compared_rect)
        # print score, compared_rect.shape, self.template.shape
        # assert score.size == 1
        if score.size != 1:
            return -888
        return float(score)

    def apply_homography(self, h):
        """
        calculate new point after homography transform
        """
        transformed = np.dot(h,[self.x, self.y, 1])
        self.x, self.y, _ = transformed/transformed[2]

    def ooi(self, frame=None):
        if frame is not None:
            return not(0 <= self.x <= frame.shape[1] and 0 <= self.y <= frame.shape[0])
        else:
            return not(0 <= self.x <= self.cols and 0 <= self.y <= self.rows)

    def template_in_frame(self, frame=None):
        """
        is template inside frame
        """
        x, y = int(self.x), int(self.y)
        dx, dy = self.TEMPLATE_WIDTH / 2, self.TEMPLATE_HIGHT / 2
        if frame is not None:
            return  dx <= x <= frame.shape[1] - dx and dy <= y <= frame.shape[0] - dy
        else:
            return dx <= x <= self.cols - dx and dy <= y <= self.rows - dy

    def occluded(self, frame):
        """
        is point not visible
        """
        assert len(frame.shape) == 2, "frame must be grayscale"
        return self.score(frame) < 0.4 # TODO: arbitrary value
        # return False

    def generate_template(self, frame):
        assert len(frame.shape) == 2, "frame must be grayscale"
        x, y = int(self.x), int(self.y)
        assert 0 <= x <= frame.shape[1] and 0 <= y <= frame.shape[0], "coordinates are out-of-image"
        dx = min(FeaturePoint.TEMPLATE_WIDTH / 2, x, frame.shape[1] - x)
        dy = min(FeaturePoint.TEMPLATE_HIGHT / 2, y, frame.shape[0] - y)
        self.template =  frame[y - dy:y + dy, x - dx:x + dx].copy()
        self.TEMPLATE_WIDTH = 2 * dx
        self.TEMPLATE_HIGHT = 2 * dy

    def __repr__(self):
        return "FeaturePoint(%f, %f)" % (self.x, self.y)


class FeaturePointManager:
    feature_params = dict(maxCorners=15,
                      qualityLevel=0.4,
                      minDistance=70,
                      blockSize=7)

    POINT_MASK_SIZE = 50
    _colors = np.random.randint(0, 255, (100, 3))
    _points = []

    # ped detection
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    detector = DetectorAPI()

    @staticmethod
    def generate_points_for_frame(frame, mask=None, consider_prev_points=False):
        """
        generate new feature points for frame
        :param consider_prev_points: prevent generation of points next to older
        points
        """
        # assert len(frame.shape) == 2, "frame must be grayscale"
        frame_color = frame
        frame = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
        if mask is None:
            mask = np.ones_like(frame, dtype='uint8') * 255
        """
        ped detection
        rects, weights = FeaturePointManager.hog.detectMultiScale(frame, winStride=(4, 4),
            padding=(8, 8), scale=1.05)
        rects = [[int(i) for i in r] for r in rects]
        for (x, y, w, h) in rects:
            print (x, y, w, h)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 0, -1)
        """

        # coco human detector
        resized = cv2.resize(frame_color, (1280, 720))
        mask_ped = np.zeros(resized.shape[:2], dtype='uint8')
        threshold = 0.7
        boxes, scores, classes, num = FeaturePointManager.detector.processFrame(resized)
        dx, dy = FeaturePoint.TEMPLATE_WIDTH/2, FeaturePoint.TEMPLATE_HIGHT/2
        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                cv2.rectangle(mask_ped,(box[1]-dx,box[0]-dy),(box[3]+dx,box[2]+dy),255,-1)
        mask_ped = cv2.resize(mask_ped, (mask.shape[1], mask.shape[0]))
        mask = cv2.subtract(mask, mask_ped)

        if consider_prev_points:
            mask_points = np.zeros_like(mask, dtype='uint8')
            y_indices, x_indices = np.indices(mask_points.shape)
            for p in FeaturePointManager._points:
                if p.ooi():
                    continue
                p = p.tuple(integer=True)
                mask_points[(np.abs(x_indices - p[0]) < FeaturePointManager.POINT_MASK_SIZE) & (np.abs(y_indices - p[1]) < FeaturePointManager.POINT_MASK_SIZE)] = 255
            mask = cv2.subtract(mask, mask_points)

        cv2.imshow("points mask", mask)
        # imshow(mask, "points mask")
        p0 = cv2.goodFeaturesToTrack(frame, mask=mask, **FeaturePointManager.feature_params)
        for p in p0:
            FeaturePointManager._points.append(FeaturePoint(p[0,0], p[0, 1], frame))

    @staticmethod
    def get_visible_points(frame=None):
        """
        return all visible points in a vector shaped (n,1,2)
        n - num of points
        """
        visible_points = [[[p.x, p.y]] for p in FeaturePointManager._points
                          if not p.ooi() and not (frame is not None and p.occluded(frame))]
        return np.array(visible_points, dtype='float32')

    @staticmethod
    def update_visible_points(new, status_vec, frame):
        assert len(frame.shape) == 2, "frame must be grayscale"
        visible_points = [p for p in FeaturePointManager._points
                            if not p.ooi() and not p.occluded(frame)]
        assert len(visible_points) == new.shape[0], "vector of new locations should be same length as visible points"
        for p, new_loc, st in zip(visible_points, new, status_vec):
            if st == 0:
                FeaturePointManager._points.remove(p) # TODO: good idea?
            elif st == 1:
                p.x, p.y = new_loc[0]

    @staticmethod
    def apply_homography(h, frame, only_non_visible_points=False):
        """
        apply homography on all points
        TODO: how much do i lose in storing points in list?
        """
        if only_non_visible_points:
            points = (p for p in FeaturePointManager._points if p.ooi() or p.occluded(frame))
        else:
            points = FeaturePointManager._points
        for p in points:
            p.apply_homography(h)

    @staticmethod
    def draw_points_on_frame(frame):
        # TODO: rewrite
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 1
        lineType               = 2
        visible_points = [p for p in FeaturePointManager._points
                          if not p.ooi()]
        for i, point in enumerate(visible_points):
            cv2.circle(frame, point.tuple(integer=True), 3, FeaturePointManager._colors[point.ID], -1)
            cv2.putText(frame, "{:+.3f}".format(point.score(gray)),
                        point.tuple(integer=True),
                        font,
                        fontScale,
                        FeaturePointManager._colors[i],
                        lineType)


class OptFlowTracker:

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))



    def __init__(self, frame):
        """ assume frame is gray """
        assert len(frame.shape) == 3, "frame must be color"
        self.old = frame
        self.color = np.random.randint(0, 255, (100, 3))
        FeaturePointManager.generate_points_for_frame(frame)
        # self.corner_mask = np.zeros(frame.shape[:2], dtype='uint8')
        # self.corner_mask[:150, :] = 255

    def find_homography(self, frame):
        assert len(frame.shape) == 3, "frame must be color"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p0 = FeaturePointManager.get_visible_points()
        if p0.shape[0] < 4:
            FeaturePointManager.generate_points_for_frame(self.old, consider_prev_points=True)
            p0 = FeaturePointManager.get_visible_points()
        if p0.size != 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.old, frame, p0, None, **self.lk_params)
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            # FeaturePointManager.update_visible_points(p1, st, gray)
            h, mask = cv2.findHomography(good_old, good_new, cv2.RANSAC) # TODO: other methods?
        self.old = frame.copy()
        return h


def _test_feature_generation():
    import cv2
    from etc.etc import imshow
    cap = cv2.VideoCapture("../les1.mp4")
    for i in range(100):
        cap.grab()
    while True:
        ret, frame = cap.read()
        FeaturePointManager.generate_points_for_frame(frame)
        FeaturePointManager.draw_points_on_frame(frame)
        imshow(frame)
        FeaturePointManager._points = []
        for i in range(24):
            cap.grab()

def _test_find_homography():
    import cv2
    import matplotlib.pyplot as plt
    cap = cv2.VideoCapture("../les1.mp4")
    for i in range(312):
        cap.grab()
    ret, last_frame = cap.read()
    tracker = OptFlowTracker(last_frame)
    fig = plt.figure(figsize=(16,16))
    ret, frame = cap.read()
    h = tracker.find_homography(frame)
    warped = cv2.warpPerspective(last_frame, h, frame.shape[:2][::-1])
    fig.add_subplot(2, 2, 1)
    FeaturePointManager.draw_points_on_frame(last_frame)
    plt.imshow(last_frame[:,:,::-1])
    fig.add_subplot(2, 2, 2)
    FeaturePointManager.apply_homography(h, frame)
    FeaturePointManager.draw_points_on_frame(frame)
    plt.imshow(frame[:,:,::-1])
    fig.add_subplot(2,2,3)
    plt.imshow(warped[:,:,::-1])
    fig.add_subplot(2,2,4)
    plt.imshow(cv2.addWeighted(frame,.5,warped,.5,0)[:,:,::-1])
    plt.show()
    fig.clear()

if __name__ == "__main__":
    _test_find_homography()