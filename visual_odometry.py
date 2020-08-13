import cv2
import numpy as np
from sp_detector import PointTracker

def sp_feature_tracking(frame, detector, tracker):

    pts, desc, heatmap = detector.run(frame)

    # Add points and descriptors to the tracker.
    tracker.update(pts, desc)

    # Get tracks for points which were match successfully across all frames.
    tracks = tracker.get_tracks(min_length=1)

    # Normalize track scores to [0,1].
    tracks[:, 1] /= float(detector.nn_thresh)
    kp1, kp2 = tracker.draw_tracks(tracks)

    return kp1, kp2

def fast_feature_tracking(image_ref, image_cur, px_ref):

    lk_params = {'winSize': (21, 21),
                 # 'maxLevel' : 3,
                 'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)}

    kp2, st, err = cv2.calcOpticalFlowPyrLK(
        image_ref, image_cur, px_ref, None, **lk_params)  # shape: [k,2] [k,1] [k,1]

    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]

    return kp1, kp2

class VisualOdometry:

    def __init__(self, name, cam_params, detector, tracker):
        self.name       = name
        self.cam_params = cam_params
        self.last_frame = None
        self.last_pose  = None
        self.cur_r      = None
        self.cur_t      = None
        self.px_ref     = None
        self.px_cur     = None
        self.detector   = detector
        self.tracker    = tracker
        self.pp = (cam_params['cx'], cam_params['cy'])
        self.x_list, self.y_list = [], []

    def initialize(self, first_frame=None, second_frame=None):

        if self.name == 'fast':
            # Process first frame
            self.px_ref = self.detector.detect(first_frame)
            self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)

            # Process second frame
            self.px_ref, self.px_cur = self.tracker(first_frame, second_frame, self.px_ref)

        elif self.name == 'sp':
            self.point_tracker = PointTracker(max_length=2, nn_thresh=self.detector.nn_thresh)
            self.px_ref, self.px_cur = self.tracker(first_frame, self.detector, self.point_tracker)
            self.px_ref, self.px_cur = self.tracker(second_frame, self.detector, self.point_tracker)

        # Get the essential matrix
        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref,
                                       focal=self.cam_params['fx'],
                                       pp=self.pp,
                                       method=cv2.RANSAC,
                                       prob=0.999,
                                       threshold=1.0)

        _, self.cur_r, self.cur_t, mask = cv2.recoverPose(E,
                                                          self.px_cur,
                                                          self.px_ref,
                                                          focal=self.cam_params['fx'],
                                                          pp=self.pp)
        self.px_ref = self.px_cur
        self.last_frame = second_frame
        self.last_pose = self.cur_t.reshape(-1)

    def process_frame(self, frame):

        assert (frame.ndim == 2 and
                frame.shape[0] == self.cam_params['height'] and
                frame.shape[1] == self.cam_params['width']), "Frame: provided image has not the same size as the camera model or image is not grayscale"

        k_min_num_feature = 500

        if self.name == 'fast':
            self.px_ref, self.px_cur = self.tracker(self.last_frame, frame, self.px_ref)
        elif self.name == 'sp':
            self.px_ref, self.px_cur = self.tracker(frame, self.detector, self.point_tracker)

        E, mask = cv2.findEssentialMat(self.px_cur,
                                       self.px_ref,
                                       focal=self.cam_params['fx'],
                                       pp=self.pp,
                                       method=cv2.RANSAC,
                                       prob=0.999,
                                       threshold=1.0)

        _, R, t, mask = cv2.recoverPose(E, self.px_cur,
                                        self.px_ref,
                                        focal=self.cam_params['fx'],
                                        pp=self.pp)


        self.cur_t = self.cur_t + 0.7*self.cur_r.dot(t)
        self.cur_r = R.dot(self.cur_r)
        self.x_list.append(self.cur_t[0])
        self.y_list.append(self.cur_t[2])

        if self.name == "fast" and self.px_ref.shape[0] < k_min_num_feature:
            self.px_cur = self.detector.detect(frame)
            self.px_cur = np.array(
                [x.pt for x in self.px_cur], dtype=np.float32)

        self.px_ref = self.px_cur
        self.last_frame = frame

    def get_absolute_scale(self, frame_pose):  # specialized for KITTI odometry dataset

        x_prev, y_prev, z_prev = self.last_pose
        x, y, z = frame_pose
        self.last_pose = frame_pose
        return np.sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev))
