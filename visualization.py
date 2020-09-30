import cv2
import matplotlib.pyplot as plt

def update_visualization(frame, figure, ax, lines, trajectory_data):

    # Features
    vo_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for (u, v) in trajectory_data['features']:
        cv2.circle(vo_frame, (u, v), 3, (0, 200, 0))
    cv2.putText(vo_frame, trajectory_data['detector_name'], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (190, 170, 10), 2, cv2.LINE_AA)

    cv2.imshow("Features Detection", vo_frame)

    # Trajectory
    ## Update data (with the new and the old points)
    if 'true_x_list' in trajectory_data and 'true_z_list' in trajectory_data:
        lines['true'].set_xdata(trajectory_data['true_x_list']); lines['true'].set_ydata(trajectory_data['true_z_list'])

    lines['vo'].set_xdata(trajectory_data['vo_x_list']); lines['vo'].set_ydata(trajectory_data['vo_z_list'])

    # Need both of these in order to rescale
    ax.relim()
    ax.autoscale_view()

    # We need to draw and flush
    figure.canvas.draw()
    figure.canvas.flush_events()

def visualization_setup(vo_name):

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
    ax.set(xlabel='x', ylabel='z', title='Visual Odometry Trajectory')
    ax.legend('True Pose')

    true_lines, = ax.plot([], [], 'o', color='blue', markersize=3, linestyle='--', label='ground truth')
    vo_lines, = ax.plot([], [], 'o', color='orange', markersize=3, linestyle='--', label=vo_name)

    leg = ax.legend(loc='upper right', fancybox=True, shadow=True)
    leg.get_frame().set_alpha(0.4)

    # Auto scale on unknown axis
    ax.set_autoscaley_on(True)

    # Other stuff
    ax.grid()
    lines = {'true' : true_lines, 'vo' : vo_lines}

    lined = dict()
    for leg_line, orig_line in zip(leg.get_lines(), lines.values()):
        leg_line.set_picker(10)
        lined[leg_line] = orig_line
    figure.canvas.mpl_connect('pick_event', on_pick)

    cv2.namedWindow('Features Detection', cv2.WINDOW_NORMAL)

    return figure, ax, lines

def visualize_optical_flow(cap, model):

    # Two visualizations mode, optical flow direction and optical flow magnitude.
    mode, pause = False, False
    while True:

        if not pause:
            _, img_1 = cap.read()
            _, img_2 = cap.read()

            dim = (int(img_1.shape[1] * 0.3), int(img_1.shape[0] * 0.3))

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

def visualize_motion_segmentation(outlier_mask, config, img):

    outlier_mask = outlier_mask.reshape(config['height'], config['width'])
    img_motion_segmentation = img
    overlay = img_motion_segmentation.copy()

    for y, row in enumerate(outlier_mask):
        for x, is_outlier in enumerate(row):
            if is_outlier:
                cv2.circle(overlay, (x, y), 1, (0, 0, 255), -1)

    # Add transparency to red circles
    alpha = 0.3
    img_motion_segmentation = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    resize_dim = (int(img_motion_segmentation.shape[1] * 3), int(img_motion_segmentation.shape[0] * 3))
    img_motion_segmentation = cv2.resize(img_motion_segmentation, resize_dim, interpolation=cv2.INTER_AREA)
    cv2.imshow("Motion segmentation", img_motion_segmentation)
