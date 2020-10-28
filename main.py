import argparse
import cv2
import glob
import sys
import yaml
import numpy as np
import torch

sys.path.append('models/raft/')
from visualization import visualization_setup, update_visualization
from visual_odometry import initialize_vo, get_scale, get_frame
from models.super_glue.utils import frame2tensor
from motion_detection import motion_probability
from raft import RAFT

import matplotlib
matplotlib.use('TkAgg')

def initialize(args):

    # Load configuration file and set camera params
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['cam_params'] = config['cameras'][args.camera_id]
    config['cam_centers'] = (config['cam_params']['cx'], config['cam_params']['cy'])

    # Get ground truth poses
    if args.dataset == 'kitti':
        with open(args.poses_path, 'r') as f:
            true_poses = np.array([float(p) for pose in f.read().split('\n') for p in pose.split()])
            true_poses = true_poses.reshape(-1, 12)
            true_poses = [[true_pose[3], true_pose[7], true_pose[11]] for true_pose in true_poses]
    elif args.dataset == 'tum':
        with open(args.poses_path, 'r') as f:
            true_poses = np.array([float(p) for pose in f.read().split('\n') for p in pose.split() if '#' not in pose])
            true_poses = true_poses.reshape(-1, 8)
            true_poses = [[true_pose[1], true_pose[2], true_pose[3]] for true_pose in true_poses]
    else:
        true_poses = None

    # Get image loader (images from camera or folder)
    if args.images_path is not None:
        image_getter = (cv2.imread(image_path) for image_path in sorted(glob.glob(f'{args.images_path}/*')))
    elif args.camera_source is not None:
        video_cap = cv2.VideoCapture(args.camera_source)
        image_getter = video_cap
    else:
        print('Error, image source is not defined')
        sys.exit(1)

    torch.set_grad_enabled(False)

    # RAFT
    optical_flow_model = torch.nn.DataParallel(RAFT(args))
    optical_flow_model.load_state_dict(torch.load(args.optical_flow_model))
    model = optical_flow_model.module
    model.to(config['device'])
    model.eval()

    return true_poses, image_getter, config, optical_flow_model

def main():

    parser = argparse.ArgumentParser(description='Monocular Visual Odometry with Dynamic Objects Detection')
    parser.add_argument('--images_path',         type=str, help='Path to folder with images',                     default=None)
    parser.add_argument('--dataset',             type=str, help='Dataset name (kitti or tum)',                    default=None)
    parser.add_argument('--poses_path',          type=str, help='Path to file with pose',                         default=None)
    parser.add_argument('--config_path',         type=str, help='Path to yaml file with vo params',               default='config.yaml')
    parser.add_argument('--camera_source',       type=int, help='Camera source id to get images',                 default=None)
    parser.add_argument('--camera_id',           type=str, help='ID of the camera in cameras params yaml file',   required=True)
    parser.add_argument('--results_path',        type=str, help='Path to folder to save visual odometry results', default='results')
    parser.add_argument('--optical_flow_model',            help="restore checkpoint",                             default='models/raft/raft-things.pth',)
    parser.add_argument('--small',                         help='use small model for optical flow',               action='store_true',)
    parser.add_argument('--mixed_precision',               help='use mixed precision',                            action='store_true',)
    parser.add_argument('--alternate_corr',                help='use efficient correlation implementation',       action='store_true',)
    parser.add_argument('--debug',                         help='Activate / deactivate debug',                    default=True)
    args = parser.parse_args()

    true_poses, image_getter, config, optical_flow_model = initialize(args)
    cur_R, cur_T, matching, reference, keys, features, dynamic_model = initialize_vo(image_getter, config)

    # Error and trajectory visualization variables
    figure, ax, lines = visualization_setup()

    trajectory_data = {'step': [], 'error_x': [], 'error_y': [], 'error_z': []}
    if true_poses is not None:
        true_poses_gen = (true_pose for true_pose in true_poses)
        prev_true_pose = next(true_poses_gen); cur_true_pose = next(true_poses_gen)
        scale = get_scale(prev_true_pose, cur_true_pose)
        cur_T = scale * cur_T
        trajectory_data.update({'true_x_list': [prev_true_pose[0], cur_true_pose[0]], 'true_z_list': [prev_true_pose[2], cur_true_pose[2]]})

    cur_T += list(map(int, config['visual_odometry']['origin_offset'].split()))
    trajectory_data.update({'detector_name': 'Super Glue',
                            'vo_x_list': [0, cur_T[0]],
                            'vo_z_list': [0, cur_T[2]],
                            'features': features})

    i = 0
    while True:
        frame = get_frame(image_getter)
        frame = cv2.resize(frame, (config['width'], config['height']), interpolation=cv2.INTER_AREA)
        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_tensor = frame2tensor(frame_grey, config['device'])
        current = matching({**reference, 'image1': frame_tensor})
        current['frame1'] = frame
        kpts0 = reference['keypoints0'][0].cpu().numpy()
        kpts1 = current['keypoints1'][0].cpu().numpy()
        matches = current['matches0'][0].cpu().numpy()
        confidence = current['matching_scores0'][0].cpu().numpy()
        valid_matches = matches > -1
        valid_confidence = confidence > 0.95
        valid = np.logical_and(valid_matches, valid_confidence)
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        fg_probability = motion_probability(reference['frame0'], current['frame1'], optical_flow_model, dynamic_model, config)

        static_pixels_index = [i for i, p in enumerate(mkpts0) if fg_probability[int(p[1])][int(p[0])]]
        mkpts0 = mkpts0[static_pixels_index]
        mkpts1 = mkpts1[static_pixels_index]

        E, mask = cv2.findEssentialMat(mkpts1, mkpts0, focal=config['cam_params']['fx'], pp=config['cam_centers'], method=cv2.RANSAC, prob=0.999, threshold=0.5)
        _, R, T, mask = cv2.recoverPose(E, mkpts1, mkpts0, focal=config['cam_params']['fx'], pp=config['cam_centers'])
        T = T.reshape(-1)

        if true_poses is not None:
            cur_true_pose = next(true_poses_gen)
            trajectory_data['true_x_list'].append(cur_true_pose[0]); trajectory_data['true_z_list'].append(cur_true_pose[2])

            scale = get_scale(prev_true_pose, cur_true_pose)
            if scale > 0.1:
                cur_T = cur_T + scale * cur_R.dot(T)
                cur_R = cur_R.dot(R)

            # Error calculation
            vo_x, vo_y, vo_z = cur_T
            true_x, true_y, true_z = cur_true_pose

            trajectory_data['step'].append(i)
            trajectory_data['error_x'].append(np.subtract(vo_x, true_x))
            trajectory_data['error_y'].append(np.subtract(vo_y, true_y))
            trajectory_data['error_z'].append(np.subtract(vo_z, true_z))

            prev_true_pose = cur_true_pose

        else:
            cur_T = cur_T + cur_R.dot(T)
            cur_R = R.dot(cur_R)

        trajectory_data['vo_x_list'].append(vo_x)
        trajectory_data['vo_z_list'].append(vo_z)
        update_visualization(reference['frame0'], figure, ax, lines, trajectory_data, fg_probability)
        i += 1

        reference = {k + '0': current[k + '1'] for k in keys}
        reference['image0'] = frame_tensor
        reference['frame0'] = frame
        trajectory_data['features'] = mkpts1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)
    if isinstance(image_getter, cv2.VideoCapture):
        image_getter.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
