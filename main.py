import argparse
import yaml
import cv2
import numpy as np

import visual_odometry as vo


def initialize():
    parser = argparse.ArgumentParser(description='Visual Odometry.')
    parser.add_argument('--images_folder_path', type=str, help='Path to folder with images.')
    parser.add_argument('--images_camera_source', type=str, help='Camera ID to get images.')
    parser.add_argument('--poses_file_path', type=str, help='Path to file with ground truth/poses.')
    parser.add_argument('--cameras_params_path', type=str, help='Path to file with cameras parameters.',
                        default='cameras_params.yaml')

    args = parser.parse_args()
    images_folder_path = args.images_path
    images_camera_source = args.images_camera_source
    poses_file_path = args.poses_path

    with open(args.cameras_params_path, 'r') as f:
        cameras_params = yaml.safe_load(f)

    return images_folder_path, images_camera_source, poses_file_path , cameras_params


def main():
    images_path, poses_path, cameras_params = initialize()

    vo.VisualOdometry


if __name__ == '__main__':
    main()
