import cv2
import os
import time

MAIN = (0, 0, 255)


class Tracker:
    def __init__(self, path, showResult=False, saveData=True, border=5):
        self.tracker = cv2.TrackerCSRT_create()
        self.path = path
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        self.show = showResult
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        self.border = border
        self.isSave = saveData

    def update(self, image, objectID, bbox):
        height, width, depth = image.shape
        start = time.time()

        if bbox is None:
            return image, bbox

        success, box = self.tracker.update(image)
        if success:
            bbox = [int(v) for v in box]
            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
            if self.isSave:
                self.save(objectID, image, [ymin - self.border, ymax + self.border, xmin - self.border, xmax + self.border])

            if self.show:
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), MAIN, 1)
                cv2.putText(image, 'ID #{}'.format(objectID), (xmin, ymin - 10), self.font, 0.75, MAIN)

        duration = time.time() - start
        cv2.putText(image, 'FPS {:.2f}'.format(1 / duration), (5, height - 5), self.font, 0.75, MAIN)
        return image, bbox

    def init(self, image, bbox):
        self.tracker.init(image, bbox)

    def save(self, objectID, image, bbox):
        crop = image[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        name = '{}-{}.jpg'.format(str(objectID), str(time.time()))
        cv2.imwrite(os.path.join(self.path, name), crop)
