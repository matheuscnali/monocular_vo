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
    if 'true_x_list' in trajectory_data and 'true_y_list' in trajectory_data:
        lines['true'].set_xdata(trajectory_data['true_x_list']); lines['true'].set_ydata(trajectory_data['true_y_list'])

    lines['vo'].set_xdata(trajectory_data['vo_x_list']); lines['vo'].set_ydata(trajectory_data['vo_y_list'])

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
    ax.set(xlabel='x', ylabel='y', title='Visual Odometry Trajectory')
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
