from matplotlib import pyplot as plt
import cv2
import numpy as np

DEFAULT_FONT_KWARGS = {
    "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
    "org": (10, 500),
    "fontScale": 1,
    "color": (255, 255, 255),
    "thickness": 2,
}

imshow_id = 0
def imshow(img, title=None, size=8, gray=False):
    'display opencv image'
    global imshow_id
    if title is None:
        title = str(imshow_id)
        imshow_id += 1
    fig = plt.figure(figsize=(size, size))
    gray = len(img.shape) == 2
    if gray:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img[:, :, ::-1])
    plt.title(title + "\n" + str(img.dtype))
    plt.show()


def color_hist(img):
    'plot histogram of colors'
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()

def alignImages(im1, im2, max_features=500, good_match_percent=0.15, stitch=False):
    """
    warps im1 to im2 and returns homography
    """
    # Convert images to grayscale
    if len(im1.shape) != 2:
        im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    else:
        im1Gray = im1
    if len(im2.shape) != 2:
        im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    else:
        im2Gray = im2

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * good_match_percent)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    # cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    if stitch:
        height, width = im2.shape[0], im1.shape[1] + im2.shape[1]
        im1Reg = cv2.warpPerspective(im1, h, (width, height))
        im1Reg[0:im2.shape[0], 0:im2.shape[1]] = im2
    else:
        height, width = im2.shape[0], im2.shape[1]
        im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h

# im = np.ones((50,50,3), dtype='uint8') * 255
# build_board(im)

def waitQ():
    'display images and exit with q'
    while True:
        k = cv2.waitKey() & 0xff
        if k == ord('q'):
            break
    cv2.destroyAllWindows()

def diff(curr, last):
    """
    returns binarized diff between two frames and number
    of different pixels.
    """
    THRESH = 100
    diff_img = cv2.absdiff(curr, last)
    _, diff_img = cv2.threshold(diff_img, THRESH, 255, cv2.THRESH_BINARY)
    return diff_img, int(np.sum(diff_img) / 255)

def isRotating(curr, last, display=False):
    """
    true if difference between two frames is caused by rotation
    """
    ROTATION_THRESHOLD = 20000
    diff_im, diff_sum = diff(curr, last)
    if display:
        cv2.imshow('diff', diff_im)
    return diff_sum > ROTATION_THRESHOLD

"""
def _rectifyBoard(im):
    coords = build_board(im)
    pts1 = np.float32(coords)
    pts2 = np.float32([(0, 300), (0, 0), (750, 0), (750, 300)])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    return cv2.warpPerspective(im,M,(750,300))
"""

if __name__ == "__main__":
    # im1 = cv2.imread("../snaps/13.png", cv2.IMREAD_GRAYSCALE)
    # b1 = _rectifyBoard(im1)
    # cv2.imwrite("../b1.png", b1)
    b1 = cv2.imread("../b1.png", cv2.IMREAD_GRAYSCALE)
    # im2 = cv2.imread("../snaps/20.png", cv2.IMREAD_GRAYSCALE)
    # b2 = _rectifyBoard(im2)
    # cv2.imwrite("../b2.png", b2)
    b2 = cv2.imread("../b2.png", cv2.IMREAD_GRAYSCALE)
