import argparse
import bob.ip.optflow.liu
import cv2
import time
import sys
import yaml
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def visualize_motion_segmentation(outlier_mask, img):

    outlier_mask = outlier_mask.reshape(img.shape)
    img_motion_segmentation = cv2.cvtColor(img_2, cv2.COLOR_GRAY2BGR)
    overlay = img_motion_segmentation.copy()

    for y, row in enumerate(outlier_mask):
        for x, is_outlier in enumerate(row):
            if is_outlier:
                cv2.circle(overlay, (x, y), 1, (0, 0, 255), -1)

    # Add transparency to red circles
    alpha = 0.3
    img_motion_segmentation = cv2.addWeighted(overlay, alpha, img_motion_segmentation, 1 - alpha, 0)
    resize_dim = (int(img_motion_segmentation.shape[1] * 3), int(img_motion_segmentation.shape[0] * 3))
    img_motion_segmentation = cv2.resize(img_motion_segmentation, resize_dim, interpolation=cv2.INTER_AREA)
    cv2.imshow("Motion segmentation", img_motion_segmentation)

def visualize_optical_flow(optical_flow, cap, optical_flow_kwargs):

    # Two visualizations mode, optical flow direction and optical flow magnitude.
    mode, pause = False, False
    while True:

        if not pause:
            _, img_1 = cap.read(); img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            _, img_2 = cap.read(); img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

            dim = (int(img_1.shape[1] * 0.3), int(img_1.shape[0] * 0.3))

            img_1 = cv2.resize(img_1, dim, interpolation=cv2.INTER_AREA)
            img_2 = cv2.resize(img_2, dim, interpolation=cv2.INTER_AREA)

            if optical_flow == 'farnelback':
                flow = cv2.calcOpticalFlowFarneback(img_1, img_2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                v_x = flow[:, :, 0].flatten()
                v_y = flow[:, :, 1].flatten()
            elif optical_flow == 'ce liu':
                (u, v, warped) = bob.ip.optflow.liu.sor.flow(img_1, img_2, **optical_flow_kwargs)
                v_x = u.flatten()
                v_y = v.flatten()
                flow = [[x, y] for x, y in zip(v_x, v_y)]
                flow = np.array(flow).reshape(img_1.shape[0], img_1.shape[1], 2)

            if mode:
                img_with_flow = draw_flow(img_2, flow)
                img_with_flow = cv2.resize(img_with_flow, (dim[0] * 3, dim[1] * 3), interpolation=cv2.INTER_AREA)
                cv2.imshow(f"{optical_flow} optical flow", img_with_flow)
            else:
                v_mag = np.sqrt(np.add(np.power(v_x, 2), np.power(v_y, 2)))
                v_mag = v_mag.reshape(img_2.shape)
                v_mag = cv2.resize(v_mag, (dim[0] * 3, dim[1] * 3), interpolation=cv2.INTER_AREA)
                cv2.imshow(f"{optical_flow} optical flow", v_mag)
        else:
            time.sleep(0.1)

        k = cv2.waitKey(1)

        if k == ord('m'):
            mode = not mode
        elif k == ord('p'):
            pause = not pause
        elif k == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

def draw_flow(img, flow, step=8, color=(0, 255, 0)):

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, color)
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, color, -1)
    return vis

def activation_function(x):

    fg_probability = 2 / (1 + np.exp(-2 * (x))) - 1
    return fg_probability

def get_motion_vector(img_1, img_2, optical_flow_kwargs):

    #flow = cv2.calcOpticalFlowFarneback(img_1, img_2, None, 0.5, 3, 5, 3, 5, 1.2, 0)
    #v_x = flow[:, :, 0].flatten()
    #v_y = flow[:, :, 1].flatten()

    (u, v, warped) = bob.ip.optflow.liu.sor.flow(img_1, img_2, **optical_flow_kwargs)
    v_x = u.flatten()
    v_y = v.flatten()

    v_mag = np.sqrt(np.add(np.power(v_x, 2), np.power(v_y, 2)))
    v_ang = np.degrees(np.arctan(np.divide(v_y, v_x)))
    return np.array([v_x, v_y, v_mag, v_ang])

def get_pca_projections(data):

    data = StandardScaler().fit_transform(data.T)

    pca = PCA(n_components=1)
    pca.fit(data)
    print(f'PCA variance ratio: {pca.explained_variance_ratio_}')
    transformed_data = pca.transform(data)

    return transformed_data.flatten()

def initialize(args):

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f.read())

    #cap = cv2.VideoCapture('/media/az/420a7b37-c2b0-476d-b5a9-36622869cdb0/image_dataset/hopkins/Hopkins155WithVideosPart8/cars9/cars9.avi')
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('/media/az/420a7b37-c2b0-476d-b5a9-36622869cdb0/image_dataset/kitti/sequences/20/project.avi')

    # Define dynamic model
    w, h = int(config['w'] * config['resize_scale']) , int(config['h'] * config['resize_scale'])
    config['dim'] = (w, h)
    Y, X = np.mgrid[0: h: 1, 0: w: 1].reshape(2, -1).astype(int)
    dynamic_model = np.array([X ** 2, Y ** 2, X * Y, X, Y, X * 0 + 1]).T

    fps = []

    optical_flow_kwargs = {
        'alpha': 1,
        'ratio': 0.5,
        'min_width': 7,
        'n_outer_fp_iterations': 6,
        'n_inner_fp_iterations': 2,
        'n_sor_iterations': 20
    }

    # Check optical flow working
    visualize_optical_flow('ce liu', cap, optical_flow_kwargs)

    return fps, cap, config, X, Y, dynamic_model, optical_flow_kwargs

def motion_from_optical_flow(img_1, img_2, optical_flow_kwargs):

    # Get motion vector from optical flow and project data into the principal component
    actual_motion = get_motion_vector(img_1, img_2, optical_flow_kwargs)
    actual_motion = get_pca_projections(actual_motion)

    actual_motion_, dynamic_model_ = actual_motion, dynamic_model
    epsilon, prev_residual = config['epsilon'] + 1e-6, 0
    while epsilon > config['epsilon']:
        # Fit a dynamic model to actual motion
        coefficients, r, rank, s = np.linalg.lstsq(a=dynamic_model_, b=actual_motion_, rcond=None)
        epsilon = abs(r - prev_residual)
        prev_residual = r

        # Compute estimated motion, pixel wise motion error and foreground probability
        a, b, c, d, e, f = coefficients
        estimated_motion = np.array(a * X ** 2 + b * Y ** 2 + c * X * Y + d * X + e * Y + f).T
        pixel_wise_motion_error = np.abs(np.subtract(actual_motion, estimated_motion))
        fg_probability = activation_function(pixel_wise_motion_error)

        # Select outlier pixels
        outlier_mask = fg_probability < config['t_motion']
        outliers_index = np.where(outlier_mask == True)

        # Update models
        dynamic_model_ = dynamic_model[outliers_index]
        actual_motion_ = actual_motion[outliers_index]

    return outlier_mask

if __name__ == '__main__':

    # Load configurations
    parser = argparse.ArgumentParser(description='Motion detector')
    parser.add_argument('--config_path', type=str, help='Path to configuration file', default='configuration/motion_detection_params.yaml')
    args = parser.parse_args()

    fps, cap, config, X, Y, dynamic_model, optical_flow_kwargs = initialize(args)

    while True:

        start = time.time()

        _, img_1 = cap.read(); img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY); img_1 = cv2.resize(img_1, config['dim'], interpolation=cv2.INTER_AREA)
        _, img_2 = cap.read(); img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY); img_2 = cv2.resize(img_2, config['dim'], interpolation=cv2.INTER_AREA)

        # Run optical flow motion segmentation
        outlier_mask = motion_from_optical_flow

        # Visualize motion segmentation
        visualize_motion_segmentation(outlier_mask, img_2)

        # Prints performance information
        elapsed_time = time.time() - start
        fps.append(1 / elapsed_time)
        print(f'FPS: {fps[-1]:.1f}\nFPS Avg: {np.mean(fps):.1f}\n')

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)






"""optical_flow_kwargs = {
    'alpha':  np.random.randint(1, 5),
    'ratio': np.random.uniform(0.5, 0.9),
    'min_width': np.random.randint(1, 18),
    'n_outer_fp_iterations': np.random.randint(1, 10),
    'n_inner_fp_iterations': np.random.randint(1, 5),
    'n_sor_iterations': np.random.randint(1, 50)
}
"""
#print(f'Selected interval time %: {int((1 - (elapsed_time - (T2_ - T1_)) / elapsed_time) * 100)}%')
