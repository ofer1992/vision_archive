import cv2
import numpy as np
import sys
print sys.path
from utils.etc import alignImages


class Panorama:
    _stitcher = cv2.createStitcher(False)

    def __init__(self, frame):
        self.panorama = frame.copy()
        self.origin = np.array([0, 0])

    def update(self, frame):
        # aligned, h = alignImages(frame, self.panorama, titch=True)
        status, aligned = Panorama._stitcher.stitch((frame, self.panorama))
        print status
        if status == 0:
            self.panorama = aligned
        cv2.imshow('pan', self.panorama)

    def __getitem__(self, item):
        pass

    def __setitem__(self, key, value):
        pass

if __name__ == "__main__":
    # """
    cap = cv2.VideoCapture('../les1.mp4')
    for i in range(110): cap.grab()
    # _, frame = cap.read()
    # pan = Panorama(frame)
    # panorama = cv2.imread("../panorama.png")
    panorama = cv2.imread("../panorama_unwarped2.png")
    while cap.isOpened():
        for i in range(50):
            cap.grab()
        ret, frame = cap.read()
        if not ret:
            break
        aligned, h = alignImages(frame, panorama)
        weighted =  cv2.addWeighted(aligned, 0.5, panorama, 0.5, 0)
        weighted = cv2.resize(weighted, (0,0), fx=0.5, fy=0.5)
        cv2.imshow('aligned', weighted)
        if cv2.waitKey(100) & 0xff == ord('q'):
            break
    # """
    """
    import os
    directory = "/home/tomer/git/vision_project/snaps/"
    l = [121, 43, 9, 2, 1]
    # files = [str(i)+".png" for i in (43, 2)]
    files = [str(i)+".png" for i in l]
    images = [cv2.imread(directory+file) for file in files]
    stitcher = cv2.createStitcher()
    stat, pano = stitcher.stitch(images)
    # for file in files[1:]:
    #     frame = cv2.imread(directory+file)
    #     cv2.imshow('im', frame)
    #     pan.update(frame)
    #     if cv2.waitKey(100) & 0xff == ord('q'):
    #         break

    cv2.imshow('../panorama.png', pano)
    cv2.imwrite('../panoramafull.png', pano)
    while cv2.waitKey(1000) != ord('q'): continue
    cv2.destroyAllWindows()
    """
    """
    import os
    directory = "/home/tomer/git/vision_project/snaps/"
    files = sorted(os.listdir(directory), key=lambda x: int(x[:x.rindex('.')]))
    images = (cv2.imread(directory+file) for file in files)
    panorama = cv2.imread("../panorama_unwarped.png")
    for frame in images:
        aligned, h = alignImages(frame, panorama)
        weighted =  cv2.addWeighted(aligned, 0.5, panorama, 0.5, 0)
        weighted = cv2.resize(weighted, (0,0), fx=0.5, fy=0.5)
        cv2.imshow('aligned', weighted)
        if cv2.waitKey(100) & 0xff == ord('q'):
            break
    """
    """
    im45, im2 = cv2.imread("../snaps/45.png"), cv2.imread("../snaps/2.png")
    im9 = cv2.imread("../snaps/14.png")
    aligned, h = alignImages(im2, im9, stitch=True)
    # aligned, h = alignImages(im9, im45, stitch=True)
    cv2.imshow('aligned', aligned)
    # cv2.imwrite('../panorama_unwarped.png', aligned)
    while cv2.waitKey() != ord('q'): continue
    """
