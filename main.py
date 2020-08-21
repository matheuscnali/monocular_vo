import argparse
import cv2
import glob
import sys
import yaml
import numpy as np

from visualization import visualization_setup, update_visualization
from visual_odometry import Detector, Tracker, initialize_vo, get_scale, validate_frame

## KITTI dataset sequences
# cam_x = 00, 01, 02
# cam_y = 03
# cam_z = 04, 05, 06, 07, 08, 09, 10

class ImageLoader:

    def __init__(self, source_type, source):
        self.source_type = source_type
        self.source = source

    def get_frame(self):
        if self.source_type == 'camera':
            ret, frame = self.source.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                print('Error getting image from camera')
                sys.exit(1)
        elif self.source_type == 'folder':
            return next(self.source)
        else:
            print(f'Error, ImageLoader get method not defined for the source {self.source_type}')
            sys.exit(1)

    def __exit__(self):
        print('yooooo!')
        self.source.release()
        cv2.destroyAllWindows()

def process_args(args):

    # Get visual odometry parameters
    vo_params = {'detector_name': args.detector_name}

    if args.cameras_params_path is not None:
        with open(args.cameras_params_path, 'r') as f:
            vo_params['cam_params'] = yaml.safe_load(f)[args.camera_id]
            vo_params['cam_centers'] = (vo_params['cam_params']['cx'], vo_params['cam_params']['cy'])

    with open(args.vo_configuration_path, 'r') as f:
        vo_params.update(yaml.safe_load(f)[args.detector_name])

    # Get ground truth poses
    if args.poses_file_path is not None:
        with open(args.poses_file_path, 'r') as f:
            # Maybe would be better change from float64 to float32
            true_poses = np.array([float(p) for pose in f.read().split('\n') for p in pose.split()])
            true_poses = true_poses.reshape(-1, 12)
            true_poses = [[true_pose[3], true_pose[7], true_pose[11]] for true_pose in true_poses]
    else:
        true_poses = None

    # Get image loader (images from camera or folder)
    if args.images_folder_path is not None:
        image_generator = (cv2.imread(image_path, 0) for image_path in sorted(glob.glob(f'{args.images_folder_path}/*')))
        image_loader = ImageLoader('folder', image_generator)
    elif args.camera_source is not None:
        video_cap = cv2.VideoCapture(args.camera_source)
        image_loader = ImageLoader('camera', video_cap)
    else:
        print('Error, image source is not defined')
        sys.exit(1)

    # Get detector and tracker
    detector = Detector(vo_params['detector_params'])
    tracker = Tracker(vo_params['tracker_params'])

    return vo_params, true_poses, image_loader, detector, tracker

def main():

    parser = argparse.ArgumentParser(description='Monocular Visual Odometry')
    parser.add_argument('--images_folder_path',    type=str, help='Path to folder with images',              default=None)
    parser.add_argument('--camera_source',         type=int, help='Camera ID to get images',                 default=None)
    parser.add_argument('--poses_file_path',       type=str, help='Path to file with ground truth/poses',    default=None)
    parser.add_argument('--camera_id',             type=str, help='ID of the camera used to provide images', default=None,                  required=True)
    parser.add_argument('--cameras_params_path',   type=str, help='Path to file with cameras params',        default='cameras_params.yaml', required=True)
    parser.add_argument('--vo_configuration_path', type=str, help='Path to file with vo params',             default='vo_params.yaml',      required=True)
    parser.add_argument('--detector_name',         type=str, help='Detector name, sp or fast',                                              required=True)
    vo_params, true_poses, image_loader, detector, tracker = process_args(parser.parse_args())

    cur_R, cur_T = initialize_vo(image_loader, detector, tracker, vo_params)

    # Error and trajectory visualization variables
    vo_errors = []
    figure, ax, lines = visualization_setup(vo_params['detector_name'])
    trajectory_data = {'detector_name': vo_params['detector_name'],
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

        frame = image_loader.get_frame()
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
            if scale > 0.1:
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
                print(f'{vo_params["detector_name"]} average error:  {avg_vo_error:.2f} \n{"-"*40}')
            i += 1

        else:
            cur_T = cur_T + cur_R.dot(T)
            cur_R = R.dot(cur_R)

        trajectory_data['vo_x_list'].append(cur_T[0])
        trajectory_data['vo_y_list'].append(cur_T[2])
        trajectory_data['features'] = tracked_cur_features
        update_visualization(frame, figure, ax, lines, trajectory_data)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)

if __name__ == '__main__':
    main()
