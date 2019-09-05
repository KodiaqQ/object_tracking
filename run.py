import argparse
from tracking import Tracker
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to test video')
    parser.add_argument('--output', type=str, required=True, help='Path to result video', default=None)
    parser.add_argument('--path', type=str, required=True, help='Path save')
    parser.add_argument('--id', help='Object ID to save', default='defaultID')
    return parser.parse_args()


def main():
    args = parse_args()
    tracker = Tracker(path=args.path, showResult=True, border=10)
    cap = cv2.VideoCapture(args.input)
    bbox = None
    objectID = args.id

    if args.output is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1280, 720))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, dsize=(1280, 720))

        key = cv2.waitKey(1) & 0xFF

        tracked_frame, bbox = tracker.update(frame, objectID, bbox)

        cv2.imshow('image', frame)
        if args.output is not None:
            out.write(frame)

        if key == ord('s'):
            bbox = cv2.selectROI('image', frame, fromCenter=False)
            tracker.init(frame, bbox)
            print('Selected {}'.format(bbox))
            continue
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if args.output is not None:
        out.release()


if __name__ == '__main__':
    main()
