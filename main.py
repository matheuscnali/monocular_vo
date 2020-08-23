import argparse
import cv2
import glob
import sys
import yaml
import numpy as np

from visualization import visualization_setup, update_visualization
from visual_odometry import Detector, Tracker, initialize_vo, get_scale, get_frame, validate_frame

## KITTI dataset sequences
# cam_x = 00, 01, 02
# cam_y = 03
# cam_z = 04, 05, 06, 07, 08, 09, 10

def process_args(args):

    # Get visual odometry parameters
    vo_params = {}

    if args.cameras_params_path is not None:
        with open(args.cameras_params_path, 'r') as f:
            vo_params['cam_params'] = yaml.safe_load(f)[args.camera_id]
            vo_params['cam_centers'] = (vo_params['cam_params']['cx'], vo_params['cam_params']['cy'])

    with open(args.vo_config_path, 'r') as f:
        config = yaml.safe_load(f)
        vo_params.update({
            'detector_params' : config['detector_params'][args.detector_name],
            'tracker_params'  : config['tracker_params'][args.tracker_name]
        })

    # Get ground truth poses
    if args.poses_path is not None:
        if args.dataset_name == 'kitti':
            with open(args.poses_path, 'r') as f:
                # Maybe would be better change from float64 to float32
                true_poses = np.array([float(p) for pose in f.read().split('\n') for p in pose.split()])
                true_poses = true_poses.reshape(-1, 12)
                true_poses = [[true_pose[3], true_pose[7], true_pose[11]] for true_pose in true_poses]
        elif args.dataset_name == 'tum':
            with open(args.poses_path, 'r') as f:
                # Maybe would be better change from float64 to float32
                true_poses = np.array([float(p) for pose in f.read().split('\n') for p in pose.split() if '#' not in pose])
                true_poses = true_poses.reshape(-1, 8)
                true_poses = [[true_pose[1], true_pose[2], true_pose[3]] for true_pose in true_poses]
    else:
        true_poses = None

    # Get image loader (images from camera or folder)
    if args.images_path is not None:
        image_source = (cv2.imread(image_path, 0) for image_path in sorted(glob.glob(f'{args.images_path}/*')))
    elif args.camera_source is not None:
        video_cap = cv2.VideoCapture(args.camera_source)
        image_source = video_cap
    else:
        print('Error, image source is not defined')
        sys.exit(1)

    # Get detector and tracker
    detector = Detector(args.detector_name, vo_params['detector_params'])
    tracker = Tracker(args.tracker_name, vo_params['tracker_params'])

    return vo_params, true_poses, image_source, detector, tracker

def main():

    parser = argparse.ArgumentParser(description='Monocular Visual Odometry')
    parser.add_argument('--dataset_name',        type=str, help='Dataset name (kitti or tum)',                  default=None)
    parser.add_argument('--images_path',         type=str, help='Path to folder with images',                   default=None)
    parser.add_argument('--poses_path',          type=str, help='Path to file with ground truth poses',         default=None)
    parser.add_argument('--vo_config_path',      type=str, help='Path to yaml file with vo params',             default='vo_params.yaml')
    parser.add_argument('--cameras_params_path', type=str, help='Path to yaml file with cameras params',        default='cameras_params.yaml')
    parser.add_argument('--camera_source',       type=int, help='Camera source id to get images',               default=None)
    parser.add_argument('--camera_id',           type=str, help='ID of the camera in cameras params yaml file', default=None, required=True)
    parser.add_argument('--detector_name',       type=str, help='Detector name, sp or fast',                                  required=True)
    parser.add_argument('--tracker_name',        type=str, help='Tracker name, optical flow or nearest neighbor',             required=True)
    vo_params, true_poses, image_source, detector, tracker = process_args(parser.parse_args())

    cur_R, cur_T = initialize_vo(image_source, detector, tracker, vo_params)

    # Error and trajectory visualization variables
    vo_errors = []
    figure, ax, lines = visualization_setup(detector.name)
    trajectory_data = {'detector_name': detector.name,
                       'vo_x_list': [cur_T[0]],
                       'vo_y_list': [cur_T[2]],
                       'features': None}

    if true_poses is not None:
        true_poses_gen = (true_pose for true_pose in true_poses)
        prev_true_pose = next(true_poses_gen); cur_true_pose = next(true_poses_gen)
        scale = get_scale(prev_true_pose, cur_true_pose)
        cur_T = scale * cur_T
        trajectory_data.update({'true_x_list': [], 'true_y_list': []})

    i = 0
    while True:

        frame = get_frame(image_source)
        validate_frame(frame, vo_params['cam_params'])

        features = detector.run(frame)
        tracked_prev_features, tracked_cur_features = tracker.run(frame, features)

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

        if true_poses is not None:
            cur_true_pose = next(true_poses_gen)
            trajectory_data['true_x_list'].append(cur_true_pose[0]); trajectory_data['true_y_list'].append(cur_true_pose[2])

            scale = get_scale(prev_true_pose, cur_true_pose)
            #if scale > 0.1:
            cur_T = cur_T + scale * cur_R.dot(T)
            cur_R = R.dot(cur_R)

            # Error calculation
            vo_x, vo_y, vo_z = cur_T
            true_x, true_y, true_z = cur_true_pose

            vo_error = np.linalg.norm(np.subtract([vo_x, vo_z], [true_x, true_z]))
            vo_errors.append(vo_error)

            avg_vo_error = np.mean(vo_errors)

            prev_true_pose = cur_true_pose

            if i % 5 == 0:
                print(f'frame number: {i}')
                print(f'{detector.name} average error:  {avg_vo_error:.2f} \n{"-"*40}')

        else:
            cur_T = cur_T + cur_R.dot(T)
            cur_R = R.dot(cur_R)

        trajectory_data['vo_x_list'].append(cur_T[0])
        trajectory_data['vo_y_list'].append(cur_T[2])
        trajectory_data['features'] = tracked_cur_features
        update_visualization(frame, figure, ax, lines, trajectory_data)
        i += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)
    if isinstance(image_source, cv2.VideoCapture):
        image_source.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
