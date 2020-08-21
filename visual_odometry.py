import cv2
import sys
import numpy as np
from sp_detector import SuperPointFrontend, PointTracker


class Detector:

    def __init__(self, detector_params):

        self.name = detector_params['name']

        if detector_params['name'] == 'sp':
            self.detector = SuperPointFrontend(**detector_params['configuration'])
        elif detector_params['name'] == 'fast':
            self.detector = cv2.FastFeatureDetector_create(**detector_params['configuration'])
        else:
            print(f"Error, detector {detector_params['name']} is not defined")
            sys.exit(1)

    def run(self, frame):

        if self.name == 'sp':
            return self.detector.run(frame)

        elif self.name == 'fast':
            cur_features = self.detector.detect(frame)
            return np.array([x.pt for x in cur_features], dtype=np.float32)

class Tracker:

    def __init__(self, tracker_params):

        self.name = tracker_params['name']

        if tracker_params['name'] == 'nearest_neighbor':
            self.tracker = PointTracker(**tracker_params['configuration'])

        elif tracker_params['name'] == 'optical_flow':
            self.tracker = cv2.calcOpticalFlowPyrLK
            self.k_min_num_feature = tracker_params['configuration']['k_min_num_feature']
            self.prev_features = None
            self.prev_frame = None
            self.lk_params = {'winSize'  : tuple(tracker_params['configuration']['win_size']),
                              'maxLevel' : tracker_params['configuration']['max_level'],
                              'criteria' : (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)}

        else:
            print(f"Error, detector {tracker_params['name']} is not defined")
            sys.exit(1)

    def run(self, cur_frame, features):

        if self.name == 'nearest_neighbor':
            pts, desc, heatmap = features

            # Add points and descriptors to the tracker.
            self.tracker.update(pts, desc)

            # Get tracks for points which were match successfully across all frames.
            tracks = self.tracker.get_tracks(min_length=1)

            # Normalize track scores to [0,1].
            tracks[:, 1] /= float(self.tracker.nn_thresh)
            prev_features, cur_features = self.tracker.draw_tracks(tracks)

            return prev_features, cur_features

        elif self.name == 'optical_flow':

            cur_features, st, err = self.tracker(self.prev_frame, cur_frame, self.prev_features, None, **self.lk_params)

            st = st.reshape(st.shape[0])

            prev_features = self.prev_features[st == 1]
            if prev_features.shape[0] < self.k_min_num_feature:
                self.prev_features = features
            else:
                self.prev_features = cur_features

            self.prev_frame = cur_frame

            cur_features = cur_features[st == 1]

            return prev_features, cur_features

def validate_frame(frame, cam_params):

    assert (frame.ndim == 2 and
            frame.shape[0] == cam_params['height'] and
            frame.shape[1] == cam_params['width']), "Frame: provided image has not the same size as the camera model or image is not grayscale"


def get_scale(prev_pose, cur_pose):
    """ The scale factor is computed from ground truth """

    x_ref, y_ref, z_ref = prev_pose
    x_cur, y_cur, z_cur = cur_pose
    return np.sqrt((x_cur - x_ref) ** 2 + (y_cur - y_ref) ** 2 + (z_cur - z_ref) ** 2)

def initialize_vo(image_loader, detector, tracker, vo_params):

    first_frame = image_loader.get_frame()
    validate_frame(first_frame, vo_params['cam_params'])
    first_features = detector.run(first_frame)

    if tracker.name == 'nearest_neighbor':
        tracker.run(first_frame, first_features)
    elif tracker.name == 'optical_flow':
        tracker.prev_features = first_features
        tracker.prev_frame = first_frame

    second_frame = image_loader.get_frame()
    validate_frame(second_frame, vo_params['cam_params'])
    second_features = detector.run(second_frame)

    tracked_cur_features, tracked_prev_features = tracker.run(second_frame, second_features)

    E, mask = cv2.findEssentialMat(tracked_cur_features, tracked_prev_features,
                                   focal=vo_params['cam_params']['fx'],
                                   pp=vo_params['cam_centers'],
                                   method=cv2.RANSAC,
                                   prob=0.999,
                                   threshold=1.0)

    _, R, T, mask = cv2.recoverPose(E, tracked_cur_features, tracked_prev_features,
                                    focal=vo_params['cam_params']['fx'],
                                    pp=vo_params['cam_centers'])

    T = T.reshape(-1)

    return R, T
