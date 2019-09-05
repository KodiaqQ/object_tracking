import argparse
from tracking import Tracker
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to test video')
    parser.add_argument('--tracker', type=str, default='mosse',
                        choices=['mosse', 'boosting', 'mil', 'kcf', 'tld', 'medianflow', 'csrt', 'gorutn'])
    return parser.parse_args()


def main():
    args = parse_args()
    tracker = Tracker(type=args.tracker)
    cap = cv2.VideoCapture(args.input)
    bbox = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, dsize=(1280, 720))

        key = cv2.waitKey(0) & 0xFF
        if bbox is not None:
            (success, box) = tracker.update(frame)

            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

        cv2.imshow('image', frame)

        if key == ord('s'):
            bbox = cv2.selectROI('image', frame, fromCenter=False)
            tracker.init(frame, bbox)
            print('Selected {}'.format(bbox))
            continue
        if key == ord('r'):
            continue
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
