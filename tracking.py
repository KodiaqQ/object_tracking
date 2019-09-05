import cv2
import numpy as np
import os

TYPES = {
    'mosse': cv2.TrackerMOSSE_create,
    'kcf': cv2.TrackerKCF_create,
    'goturn': cv2.TrackerGOTURN_create
}


class Tracker:
    def __init__(self, type):

        self.tracker = TYPES[type]()

    def update(self, image):
        result = self.tracker.update(image)
        return result

    def init(self, image, bbox):
        self.tracker.init(image, bbox)
