import cv2
import numpy as np

from utils.etc import alignImages
import matplotlib.pyplot as plt

LEFT_COLOR = (255,0,0)
RIGHT_COLOR = (0,0,255)
UNKNOWN_COLOR = (150,50,101)
PANORAMA = cv2.imread("../res/panorama.png")

CORNER_HIGHT = 50
CORNER_WIDTH = 100


class HackyBoardFinder:
    pos_to_coords = {
            0: {
                'left': np.float32([[540.0, 45.0], [550.7279663085938, 383.0291442871094], [1390.3638916015625, 431.265625], [1472.5662841796875, 55.803646087646484]] ),
                'right': None
            },
            1: {
                'left': np.float32([(127.88500209467958, 46.906786761625426), (149.25073313782997, 393.7857142857143), (1006.3935902806872, 412.63782991202345), (1016.4480519480521, 56.9612484289903)]),
                'right': np.float32([[1088.5062255859375, 62.4183349609375], [1078.1605224609375, 420.7509460449219], [2081.427978515625, 468.9205627441406], [2117.2392578125, 62.450477600097656]] )
            },
            2: {
                'left': np.float32([[-299.26141357421875, 56.18682861328125], [-271.6260681152344, 421.73199462890625], [614.5938720703125, 406.8533935546875], [608.22412109375, 56.16775894165039]] ),
                'right': np.float32([[686.8626708984375, 62.542335510253906], [684.7298583984375, 413.22930908203125], [1588.0447998046875, 421.7293701171875], [1607.1981201171875, 49.77021789550781]])
            },
            3: {
                'left': np.float32([[-809.1590576171875, 69.77044677734375], [-758.4973754882812, 468.0714111328125], [221.06240844726562, 416.015869140625], [201.5361785888672, 59.4063720703125]]),
                'right': np.float32([(281.21554252199417, 64.50209467951402), (295.04042731462096, 418.9218684541265), (1160.9809384164225, 396.2993297025555), (1178.576246334311, 44.393171344784264)]),
            },
            4: {
                'left': None,
                'right': np.float32([[-129.08318,    81.021095], [-109.32634,  440.9511], [ 759.5337, 387.65686 ], [765.97076,   45.129494]])
            }
    }
    reps = [cv2.imread("../res/snaps/"+str(i)+".png") for i in (121, 45, 9, 2, 1)]
    MAX_FEATURES = 500
    GOOD_FEATURE_THRESHOLD = 0.75
    orb = cv2.ORB_create(MAX_FEATURES)
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    rep_features = [orb.detectAndCompute(rep, None) for rep in reps]

    @staticmethod
    def _generateCorners():
        ' generates corner templates from reps '
        pos = [cv2.imread("../snaps/"+str(i)+".png") for i in (121, 45, 9, 2, 1)]
        corners_tl = [im[0:CORNER_HIGHT, 0:CORNER_WIDTH] for im in pos]
        corners_tr = [im[0:CORNER_HIGHT, -CORNER_WIDTH:] for im in pos]
        corners_bl = [im[-CORNER_HIGHT:, 0:CORNER_WIDTH] for im in pos]
        corners_br = [im[-CORNER_HIGHT:, -CORNER_WIDTH:] for im in pos]
        for i, c in enumerate(zip(corners_tl, corners_tr, corners_bl, corners_br)):
             cv2.imwrite("../corners/"+str(i)+'_tl.png', c[0])
             cv2.imwrite("../corners/"+str(i)+'_tr.png', c[1])
             cv2.imwrite("../corners/"+str(i)+'_bl.png', c[0])
             cv2.imwrite("../corners/"+str(i)+'_br.png', c[1])

    @staticmethod
    def hackyFindBoard(frame):
        """
        find boards under the assumption of 5 cam positions
        :param frame:
        :return: dict(keys=left/right, vals=board_coords),
            coords return counter-clockwise starting from top-left corner
        """
        keypoints, descriptors = HackyBoardFinder.orb.detectAndCompute(frame, None)
        match_scores = []
        for rep_keypoints, rep_descriptors in HackyBoardFinder.rep_features:
            # matches = HackyBoardFinder.matcher.match(descriptors, rep_descriptors, None)
            matches = HackyBoardFinder.matcher.knnMatch(descriptors, trainDescriptors=rep_descriptors, k=2)
            good_matches = 0
            for m in matches:
                if len(m) == 2 and m[0].distance < m[1].distance * HackyBoardFinder.GOOD_FEATURE_THRESHOLD:
                    good_matches += 1
                    # m = m[0]
                    # mkp1.append( kp1[m.queryIdx] )
                    # mkp2.append( kp2[m.trainIdx] )
            match_scores.append(good_matches)
        print match_scores
        pos = np.argmax(match_scores)
        return HackyBoardFinder.pos_to_coords[pos]


def findBoard(frame, debug=False):
    ' returns contours of boards in images '
    def epsilon(cnt):
        'epsilon function for approxPolyDp'
        return 0.01*cv2.arcLength(cnt, True)
    frame = frame.copy()
    # find contours in frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) != 2 else frame
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_edges = cv2.Canny(blurred, 100, 250, None, 3)
    kernel = np.ones((5,5),np.uint8)
    dilated_edges = cv2.dilate(canny_edges, kernel, iterations=1)
    negated = 255 - dilated_edges
    f, contours,h = cv2.findContours(negated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours.sort(key=cv2.contourArea, reverse=True)
    top3 = []
    for i, cnt in enumerate(contours[:3]):
        convex = cv2.convexHull(cnt)
        approx = cv2.approxPolyDP(convex, epsilon(convex), True)
        approx.shape = (approx.shape[0], approx.shape[2])
        # get rid of polygons touching bottom
        if np.any(frame.shape[0] - approx.T[1] < 200):
            continue
        # get rid of closet
        points_on_edge = approx[approx[:,0] == frame.shape[1]-1]
        # if len(points_on_edge) == 2:
        #     if abs(points_on_edge[0,1] - points_on_edge[1,1]) < 350:
        #         print "threw", approx, "because of closet heuristic"
        #         continue
        # can use aspect ratio, size, fact that it is far from edge
        top3.append(approx)
    colors = [UNKNOWN_COLOR] * len(top3)
    if len(top3) == 1:
      print top3
      cnt = np.float32(top3[0])
      _, h = alignImages(frame, PANORAMA)
      transformed = cv2.perspectiveTransform(cnt.reshape(cnt.shape[0], 1, cnt.shape[1]), h)
      transformed.shape = transformed.shape[0], transformed.shape[2]
      print transformed
      num_right = np.sum(transformed[:,0] > 900)
      if num_right > 2:
          colors = [RIGHT_COLOR]
      elif num_right < 3:
          colors = [LEFT_COLOR]
      # if np.any(top3[0][:,0] == 0):
      #     colors = [RIGHT_COLOR]
      # elif np.any(top3[0].T[0] == frame.shape[0]-1):
      #     colors = [LEFT_COLOR]
    if len(top3) == 2:
        if top3[0][0,0] < top3[1][0,0]:
            colors = [LEFT_COLOR, RIGHT_COLOR]
        else:
            colors = [RIGHT_COLOR, LEFT_COLOR]
    for i, approx in enumerate(top3):
        cv2.drawContours(frame, [approx], 0, colors[i], 9)
    if debug:
        cv2.imshow("Source", gray)
        cv2.imshow("canny", canny_edges)
        cv2.imshow("dilated", dilated_edges)
        cv2.imshow("negative", negated)
        cv2.imshow("after contours", frame)
        while True:
            k = cv2.waitKey() & 0xff
            if k == ord('q'):
                cv2.destroyAllWindows()
                break

    return top3, frame, colors

def rectifyBoard(im, contour):
    b_right = cv2.imread('../b1.png', cv2.COLOR_BGR2GRAY)
    b_left = cv2.imread('../b2.png', cv2.COLOR_BGR2GRAY)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    det = [box]
    corners = ['tl', 'bl', 'tr', 'br']
    hight = np.max(det[0].T[1]) - np.min(det[0].T[1])
    width = np.max(det[0].T[0]) - np.min(det[0].T[0])
    # print hight, width
    rectified_corners = {'tl':(0, 0),
                         'bl':(0,hight),
                         'tr':(width,0),
                         'br':(width,hight)}
    coords = det[0].tolist()
    board = {}
    coords.sort(key=lambda x: x[0])
    board['tl'], board['bl'] = sorted(coords[:2], key=lambda x: x[1])
    board['tr'], board['br'] = sorted(coords[2:4], key=lambda x: x[1])
    # print board

    pts1 = np.float32([board[c] for c in corners])
    pts2 = np.float32([rectified_corners[c] for c in corners])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(im,M,(width,hight))
    if  compareBoards(dst, b_left) < compareBoards(dst, b_right):
        print "left"
    else:
        print "right"


def compareBoards(b1, b2):
    hist1 = cv2.calcHist(b1,[0],None, [256], [0, 256])
    hist2 = cv2.calcHist(b2,[0],None, [256], [0, 256])
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)


def build_board(frame):
    """
    display image and click on corners to specify coordinates.
    expected order is top-left, bottom-left, top-right, bottom-right
    :param frame:
    :return:
    """
    fig = plt.figure(figsize=(20,20))
    if len(frame.shape) == 2:
        plt.imshow(frame, cmap='gray')
    else:
        plt.imshow(frame[:,:,::-1])
    coords = []
    def on_click(event, coords=coords):
        coords += [(event.xdata, event.ydata)]
        if len(coords) == 4:
            plt.close(fig)
        print coords
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    if len(coords) != 4:
        return None
    return coords

if __name__ == "__main__":
    # _generateCorners()
    # im = cv2.imread("../snaps/140.png")
    # print hackyFindBoard(im)
    # """
    cap = cv2.VideoCapture('../les1.mp4')
    colors = {'left': (230,230,250), 'right': (0,0,255)}
    rectified = {'left': np.zeros((300, 750, 3), dtype='uint8'),
                 'right': np.zeros((300,750,3), dtype='uint8')
                 }
    rectified_points = np.array([[0,0],
                                 [0,300],
                                 [750, 300],
                                 [750, 0]], dtype='float32').reshape(4,1,2)
    for i in range(110): cap.grab()
    while cap.isOpened():
        for i in range(50):
            cap.grab()
        ret, frame = cap.read()
        if not ret:
            break
        board = HackyBoardFinder.hackyFindBoard(frame)
        if board is not None:
            for side, coords in board.iteritems():
                if coords is not None:
                    H = cv2.getPerspectiveTransform(coords.reshape(4,1,2), rectified_points)
                    cv2.imshow(side, cv2.warpPerspective(frame, H, (750, 300)))

                    coords = np.int32(coords)
                    cv2.drawContours(frame,[coords],0, colors[side], -1)
        cv2.imshow('vid', frame)
        if cv2.waitKey(100) & 0xff == ord('q'):
            break
    # """
    def rep_and_rectify():
        corners = ['tl', 'bl', 'tr', 'br']
        det, im_with_contours = findBoard(im, True)
        hight = np.max(det[0].T[1]) - np.min(det[0].T[1])
        width = np.max(det[0].T[0]) - np.min(det[0].T[0])
        print hight, width
        rectified_corners = {'tl':(0, 0),
                             'bl':(0,hight),
                             'tr':(width,0),
                             'br':(width,hight)}
        coords = det[0].tolist()
        board = {}
        coords.sort(key=lambda x: x[0])
        board['tl'], board['bl'] = sorted(coords[:2], key=lambda x: x[1])
        board['tr'], board['br'] = sorted(coords[2:4], key=lambda x: x[1])
        print board

        pts1 = np.float32([board[c] for c in corners])
        pts2 = np.float32([rectified_corners[c] for c in corners])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(im,M,(width,hight))
        cv2.drawContours(im, [det[0]], 0, (255, 255, 255), -1)
        cv2.imshow('board', im)
        cv2.imshow('rectified', dst)
        cv2.imwrite('right_rep.png', dst)
        while True:
            k = cv2.waitKey() & 0xff
            if k == ord('q'):
                cv2.destroyAllWindows()
                break

    # det, _a, _b = findBoard(im, True)
    # for d in det:
    #     rectifyBoard(im, d)
    # print det
    # cv2.imshow('im', im)
    # while True:
    #     k = cv2.waitKey() & 0xff
    #     if k == ord('q'):
    #         cv2.destroyAllWindows()
    #         break

