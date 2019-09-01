import cv2
import os
import time
import argparse

OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "median_flow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}
WIDTH, HEIGHT = (1280, 720)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Path to video file', required=True, type=str)
    parser.add_argument('-t', '--tracker', help='Tracker type', required=True, type=str,
                        choices=['boosting', 'mil', 'kcf', 'csrt', 'median_flow', 'tld', 'mosse'])
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.path):
        print('{} is not a file'.format(args.path))
        return False

    tracker = OBJECT_TRACKERS[args.tracker]()

    cap = cv2.VideoCapture(args.path)
    initBB = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return False
        image = cv2.resize(frame, dsize=(WIDTH, HEIGHT))

        if initBB is not None:
            (success, box) = tracker.update(image)

            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(image, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)

            info = [
                ("Tracker", args.tracker),
                ("Success", "Yes" if success else "No"),
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(image, text, (10, HEIGHT - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('tracking', image)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            initBB = cv2.selectROI('tracking', image, fromCenter=False,
                                   showCrosshair=True)

            tracker.init(image, initBB)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == '__main__':
    main()
