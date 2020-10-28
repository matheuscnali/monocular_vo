import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import time


def update_visualization(frame, figure, ax, lines, trajectory_data, mask):

    # Features
    for (u, v) in trajectory_data['features']:
        cv2.circle(frame, (u, v), 3, (0, 200, 0))

    # Trajectory
    ## Update data (with the new and the old points)
    if 'true_x_list' in trajectory_data and 'true_z_list' in trajectory_data:
        lines['true'].set_xdata(trajectory_data['true_x_list']); lines['true'].set_ydata(trajectory_data['true_z_list'])

    lines['vo'].set_xdata(trajectory_data['vo_x_list']); lines['vo'].set_ydata(trajectory_data['vo_z_list'])

    lines['error_x'].set_xdata(trajectory_data['step']); lines['error_x'].set_ydata(trajectory_data['error_x'])
    lines['error_y'].set_xdata(trajectory_data['step']); lines['error_y'].set_ydata(trajectory_data['error_y'])
    lines['error_z'].set_xdata(trajectory_data['step']); lines['error_z'].set_ydata(trajectory_data['error_z'])

    # Need both of these in order to rescale
    ax[0].relim()
    ax[1].relim()
    ax[0].autoscale_view()
    ax[1].autoscale_view()

    # We need to draw and flush
    figure.canvas.draw()
    figure.canvas.flush_events()

    # Draw motion detection
    img_motion_segmentation = frame
    overlay = img_motion_segmentation.copy()

    for y, row in enumerate(mask):
        for x, is_outlier in enumerate(row):
            if is_outlier:
                cv2.circle(overlay, (x, y), 1, (0, 0, 255), -1)

    # Add transparency to red circles
    alpha = 0.3
    img_motion_segmentation = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.imshow("Motion segmentation", img_motion_segmentation)

def visualization_setup():

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
    figure, ax = plt.subplots(1, 2)

    ax[0].set(xlabel='x', ylabel='z', title='Visual Odometry Trajectory')
    ax[1].set(xlabel='Frame', ylabel='Error', title='Visual Odometry Trajectory Error')

    true_lines, = ax[0].plot([], [], 'o', color='blue', markersize=3, linestyle='--', label='Ground Truth')
    vo_lines, = ax[0].plot([], [], 'o', color='orange', markersize=3, linestyle='--', label='Super Glue')
    error_lines_X, = ax[1].plot([], [], 'o', color='green', markersize=3, linestyle='--', label='Error X')
    error_lines_Y, = ax[1].plot([], [], 'o', color='red', markersize=3, linestyle='--', label='Error Y')
    error_lines_Z, = ax[1].plot([], [], 'o', color='blue', markersize=3, linestyle='--', label='Error Z')

    trajectory_leg = ax[0].legend(loc='upper right', fancybox=True, shadow=True)
    trajectory_leg.get_frame().set_alpha(0.4)
    error_leg = ax[1].legend(loc='upper right', fancybox=True, shadow=True)
    error_leg.get_frame().set_alpha(0.4)

    # Auto scale on unknown axis
    ax[0].set_autoscaley_on(True)
    ax[1].set_autoscaley_on(True)

    # Other stuff
    ax[0].grid()
    ax[1].grid()

    lines = {'true': true_lines, 'vo': vo_lines, 'error_x': error_lines_X, 'error_y': error_lines_Y, 'error_z': error_lines_Z}

    lined = dict()
    for leg_line, orig_line in zip([*trajectory_leg.get_lines(), *error_leg.get_lines()], lines.values()):
        leg_line.set_pickradius(10)
        lined[leg_line] = orig_line
    figure.canvas.mpl_connect('pick_event', on_pick)

    return figure, ax, lines

def visualize_optical_flow(model):

    # Two visualizations mode, optical flow direction and optical flow magnitude.
    mode, pause = True, False
    while True:

        if not pause:
            img_1 = cv2.imread('/home/az/Documents/TCC/vo/test/000032.png')
            img_2 = cv2.imread('/home/az/Documents/TCC/vo/test/000033.png')

            dim = (int(img_1.shape[1] * 0.4), int(img_1.shape[0] * 0.4))

            img_1 = cv2.resize(img_1, dim, interpolation=cv2.INTER_AREA)
            img_2 = cv2.resize(img_2, dim, interpolation=cv2.INTER_AREA)
            h, w = img_2.shape[0], img_2.shape[1]

            img1 = np.array(img_1).astype(np.uint8)
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = np.array(img_2).astype(np.uint8)
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

            images = torch.stack([img1, img2], dim=0)
            images = images.to(DEVICE)

            padder = InputPadder(images.shape)
            images = padder.pad(images)[0]
            img1 = images[0, None]
            img2 = images[1, None]

            flow_low, flow_up = model(img1, img2, iters=20, test_mode=True)
            v_x = flow_up[0][0].cpu().data.numpy().flatten()
            v_y = flow_up[0][1].cpu().data.numpy().flatten()
            flow = np.array([flow_up.cpu().data.numpy()[0][1].T, flow_up.cpu().data.numpy()[0][0].T]).T

            if mode:
                img_with_flow = draw_flow(img_2, flow)
                img_with_flow = cv2.resize(img_with_flow, (dim[0] * 3, dim[1] * 3), interpolation=cv2.INTER_AREA)
                cv2.imshow("Optical flow", img_with_flow)
            else:
                v_mag = np.sqrt(np.add(np.power(v_x, 2), np.power(v_y, 2)))
                v_mag = v_mag.reshape(h, w)
                v_mag = cv2.resize(v_mag, (dim[0] * 3, dim[1] * 3), interpolation=cv2.INTER_AREA)
                cv2.imshow("Optical flow", v_mag)
        else:
            time.sleep(0.1)

        k = cv2.waitKey(1)

        if k == ord('m'):
            mode = not mode
        elif k == ord('p'):
            pause = not pause
        elif k == ord('q'):
            cv2.destroyAllWindows()
            break

def draw_flow(img, flow, step=8, color=(0, 255, 0)):

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    if len(img.shape) == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3:
        vis = img
    cv2.polylines(vis, lines, 0, color)
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, color, -1)
    return vis
