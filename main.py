import argparse
import cv2
import glob
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt

import visual_odometry as vo
from sp_detector import SuperPointFrontend

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
        self.source.release()
        cv2.destroyAllWindows()

def update_visualization(frame, fast_vo, sp_vo, figure, ax, lines, true_x_list=None, true_y_list=None):

    # Features
    fast_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for (u, v) in fast_vo.px_ref:
        cv2.circle(fast_frame, (u, v), 3, (0, 200, 0))
    cv2.putText(fast_frame, 'FAST', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (190, 170, 10), 2, cv2.LINE_AA)

    sp_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for (u, v) in sp_vo.px_ref:
        cv2.circle(sp_frame, (u, v), 3, (0, 200, 0))
    cv2.putText(sp_frame, 'SuperPoint', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (190, 170, 10), 2, cv2.LINE_AA)

    cv2.imshow("Features Detection", np.concatenate((fast_frame, sp_frame), axis=0))

    # Trajectory
    ## Update data (with the new and the old points)
    if true_x_list is not None and true_y_list is not None:
        lines['true'].set_xdata(true_x_list); lines['true'].set_ydata(true_y_list)

    lines['fast'].set_xdata(fast_vo.x_list); lines['fast'].set_ydata(fast_vo.y_list)
    lines['sp'].set_xdata(sp_vo.x_list); lines['sp'].set_ydata(sp_vo.y_list)

    # Need both of these in order to rescale
    ax.relim()
    ax.autoscale_view()

    # We need to draw *and* flush
    figure.canvas.draw()
    figure.canvas.flush_events()

def trajectory_visualization_setup():

    def on_pick(event):
        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility
        leg_line_ = event.artist
        orig_line_ = lined[leg_line_]
        vis = not orig_line_.get_visible()
        orig_line_.set_visible(vis)
        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled
        if vis:
            leg_line_.set_alpha(1.0)
        else:
            leg_line_.set_alpha(0.2)
        figure.canvas.draw()

    # Iterative plot
    plt.ion()

    # Set up plot
    figure, ax = plt.subplots()
    ax.set(xlabel='x', ylabel='y', title='Visual Odometry Trajectory')
    ax.legend('True Pose')

    true_lines, = ax.plot([], [], 'o', color='blue', markersize=3, linestyle='--', label='Ground Truth')
    fast_lines, = ax.plot([], [], 'o', color='purple', markersize=3, linestyle='--', label='FAST')
    sp_lines, = ax.plot([], [], 'o', color='orange', markersize=3, linestyle='--', label='SuperPoint')

    leg = ax.legend(loc='upper right', fancybox=True, shadow=True)
    leg.get_frame().set_alpha(0.4)

    # Auto scale on unknown axis
    ax.set_autoscaley_on(True)

    # Other stuff
    ax.grid()

    lines = {'true' : true_lines, 'fast' : fast_lines, 'sp' : sp_lines}

    lined = dict()
    for leg_line, orig_line in zip(leg.get_lines(), lines.values()):
        leg_line.set_picker(10)
        lined[leg_line] = orig_line
    figure.canvas.mpl_connect('pick_event', on_pick)

    cv2.namedWindow('Features Detection', cv2.WINDOW_NORMAL)

    return figure, ax, lines

def parse_process_args():

    parser = argparse.ArgumentParser(description='Monocular Visual Odometry using Super Point')
    parser.add_argument('--images_folder_path',  type=str, help='Path to folder with images',              default=None)
    parser.add_argument('--camera_source',       type=int, help='Camera ID to get images',                 default=None)
    parser.add_argument('--poses_file_path',     type=str, help='Path to file with ground truth/poses',    default=None)
    parser.add_argument('--camera_id',           type=str, help='ID of the camera used to provide images', default=None)
    parser.add_argument('--sp_weights_path',     type=str, help='Path to Superpoint weights file',         default=None)
    parser.add_argument('--cameras_params_path', type=str, help='Path to file with cameras params',        default='cameras_params.yaml')
    args = parser.parse_args()

    if args.cameras_params_path is not None:
        with open(args.cameras_params_path, 'r') as f:
            cam_params = yaml.safe_load(f)[args.camera_id]

    if args.poses_file_path is not None:
        with open(args.poses_file_path, 'r') as f:
            # Maybe would be better change from float64 to float32
            true_poses = np.array([float(p) for pose in f.read().split('\n') for p in pose.split()])
            true_poses = true_poses.reshape(-1, 12)
            true_poses = [[true_pose[3], true_pose[7], true_pose[11]] for true_pose in true_poses]
    else:
        true_poses = None

    if args.images_folder_path is not None:
        image_generator = (cv2.imread(image_path, 0) for image_path in sorted(glob.glob(f'{args.images_folder_path}/*')))
        image_loader = ImageLoader('folder', image_generator)
    elif args.camera_source is not None:
        video_cap = cv2.VideoCapture(args.camera_source)
        image_loader = ImageLoader('camera', video_cap)
    else:
        print('Error, image source is not defined')
        sys.exit(1)

    return image_loader, true_poses, cam_params, args.sp_weights_path

def main():

    image_loader, true_poses, cam_params, sp_weights_path = parse_process_args()

    # Configuring Visual Odometry with FAST descriptor
    fast_detector_kwargs = {
        'threshold'         : 50,
        'nonmaxSuppression' : True
    }

    fast_vo_kwargs = {
        'name'       : 'fast',
        'cam_params' : cam_params,
        'detector'   : cv2.FastFeatureDetector_create(**fast_detector_kwargs),
        'tracker'    : vo.fast_feature_tracking
    }

    sp_vo_kwargs = {
        'name'       : 'sp',
        'cam_params' : cam_params,
        'detector'   : SuperPointFrontend(weights_path=sp_weights_path,
                                          nms_dist=4,
                                          conf_thresh=0.015,
                                          nn_thresh=0.7,
                                          cuda=True),
        'tracker'    : vo.sp_feature_tracking
    }

    fast_vo = vo.VisualOdometry(**fast_vo_kwargs)
    sp_vo = vo.VisualOdometry(**sp_vo_kwargs)

    first_frame  = image_loader.get_frame()
    second_frame = image_loader.get_frame()
    fast_vo.initialize(first_frame, second_frame)
    sp_vo.initialize(first_frame, second_frame)

    if true_poses is not None:
        true_poses_gen = (true_pose for true_pose in true_poses)
        _ = next(true_poses_gen)
        fast_vo.last_pose = next(true_poses_gen)

    # Variables to store errors
    fast_errors, sp_errors = [], []

    # Trajectory visualization variables
    figure, ax, lines = trajectory_visualization_setup()
    true_x_list, true_y_list = [], []

    i = 0
    # for true_pose in true_poses:
    while True:

        # Frame processing
        frame = image_loader.get_frame()
        if true_poses is not None:
            true_pose = next(true_poses_gen)
            fast_vo.process_frame(frame, true_pose)
            sp_vo.process_frame(frame, true_pose)
            true_x_list.append(true_pose[0]); true_y_list.append(true_pose[2])
            update_visualization(frame, fast_vo, sp_vo, figure, ax, lines, true_x_list, true_y_list)

            # Error calculation
            fast_x, fast_y, fast_z = fast_vo.cur_t
            sp_x, sp_y, sp_z = sp_vo.cur_t
            true_x, true_y, true_z = true_pose

            true_point = [true_x, true_z]
            fast_est_point = [*fast_x, *fast_z]
            sp_est_point = [*sp_x, *sp_z]

            fast_error = np.linalg.norm(np.subtract(fast_est_point, true_point))
            sp_error = np.linalg.norm(np.subtract(sp_est_point, true_point))

            fast_errors.append(fast_error)
            sp_errors.append(sp_error)

            avg_fast_error = np.mean(fast_errors)
            avg_sp_error = np.mean(sp_errors)

            if i % 5 == 0:
                print(f'Frame number: {i}')
                print(f'FAST average error:        {avg_fast_error:.2f} \nSuperPoint average error:  {avg_sp_error:.2f} \n{"-"*40}')
            i += 1
        else:
            fast_vo.process_frame(frame)
            sp_vo.process_frame(frame)
            update_visualization(frame, fast_vo, figure, ax, lines)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)

if __name__ == '__main__':
    main()
