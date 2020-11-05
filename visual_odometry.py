import cv2
import numpy as np
from collections import Iterable

from models.super_glue.matching import Matching
from models.super_glue.utils import (AverageTimer, VideoStreamer, make_matching_plot_fast, frame2tensor)


def get_frame(image_getter):
    if isinstance(image_getter, cv2.VideoCapture):
        ret, frame = image_getter.read()
        if ret:
            return frame
        else:
            print('Error getting image from camera')
            return False
    elif isinstance(image_getter, Iterable):
        return next(image_getter)
    else:
        print("Error, can't get frame. Image source not defined.")
        return False

def get_scale(prev_pose, cur_pose):
    """ The scale factor is computed from ground truth """

    x_ref, y_ref, z_ref = prev_pose
    x_cur, y_cur, z_cur = cur_pose
    return np.sqrt((x_cur - x_ref) ** 2 + (y_cur - y_ref) ** 2 + (z_cur - z_ref) ** 2)

def initialize_vo(image_getter, config):

    matching = Matching(config).eval().to(config['device'])
    keys = ['keypoints', 'scores', 'descriptors']

    first_frame = next(image_getter)
    config['height'], config['width'] = int(first_frame.shape[0] * config['motion_detection']['resize_scale']), int(first_frame.shape[1] * config['motion_detection']['resize_scale'])

    first_frame = cv2.resize(first_frame, (config['width'], config['height']), interpolation=cv2.INTER_AREA)
    first_frame_grey = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    frame_tensor = frame2tensor(first_frame_grey, config['device'])
    reference = matching.superpoint({'image': frame_tensor})
    reference = {k + '0': reference[k] for k in keys}
    reference['image0'] = frame_tensor

    second_frame = get_frame(image_getter)
    second_frame = cv2.resize(second_frame, (config['width'], config['height']), interpolation=cv2.INTER_AREA)
    second_frame_grey = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)
    frame_tensor = frame2tensor(second_frame_grey, config['device'])
    current = matching({**reference, 'image1': frame_tensor})
    kpts0 = reference['keypoints0'][0].cpu().numpy()
    kpts1 = current['keypoints1'][0].cpu().numpy()
    matches = current['matches0'][0].cpu().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    E, mask = cv2.findEssentialMat(mkpts1, mkpts0, focal=config['cam_params']['fx'], pp=config['cam_centers'], method=cv2.RANSAC, prob=0.999, threshold=0.5)
    _, R, T, mask = cv2.recoverPose(E, mkpts1, mkpts0, focal=config['cam_params']['fx'], pp=config['cam_centers'])

    T = T.reshape(-1)

    reference = {k + '0': current[k + '1'] for k in keys}
    reference['image0'] = frame_tensor
    reference['frame0'] = second_frame

    # Define dynamic model
    y, x = np.mgrid[0: config['height']: 1, 0: config['width']: 1].reshape(2, -1).astype(int)
    dynamic_model = np.array([x ** 2, y ** 2, x * y, x, y, x * 0 + 1]).T
    config['x'] = x
    config['y'] = y

    return R, T, matching, reference, keys, mkpts1, dynamic_model
