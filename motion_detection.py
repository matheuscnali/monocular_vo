import cv2
import sys
import numpy as np
import time
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from utils.utils import InputPadder


def activation_function(x):

    fg_probability = 2 / (1 + np.exp(-2 * (x))) - 1
    return fg_probability

def get_motion_vector(optical_flow_model, img_1, img_2, config):

    img_1 = np.array(img_1).astype(np.uint8); img_1 = torch.from_numpy(img_1).permute(2, 0, 1).float()
    img_2 = np.array(img_2).astype(np.uint8); img_2 = torch.from_numpy(img_2).permute(2, 0, 1).float()

    images = torch.stack([img_1, img_2], dim=0)
    images = images.to(config['device'])

    padder = InputPadder(images.shape)
    images = padder.pad(images)[0]
    img_1 = images[0, None]
    img_2 = images[1, None]

    flow_low, flow_up = optical_flow_model(img_1, img_2, iters=20, test_mode=True)

    flow_up = padder.unpad(flow_up)

    flow_up = flow_up[0].permute(1, 2, 0).cpu().detach().numpy()
    v_x = flow_up[:, :, 0]
    v_y = flow_up[:, :, 1]

    rad = np.sqrt(np.square(v_x) + np.square(v_y))
    rad_max = np.max(rad)
    epsilon = 1e-5
    v_x = (v_x / (rad_max + epsilon)).flatten()
    v_y = (v_y / (rad_max + epsilon)).flatten()

    v_mag = np.sqrt(np.add(np.power(v_x, 2), np.power(v_y, 2)))
    v_ang = np.add(np.degrees(np.arctan(np.divide(v_y, v_x))), 360) % 360
    return np.array([v_x, v_y, v_mag, v_ang])

def get_pca_projections(data):

    data = StandardScaler().fit_transform(data.T)

    pca = PCA(n_components=1)
    pca.fit(data)
    transformed_data = pca.transform(data)

    return transformed_data.flatten()

def get_mega_pixel(img, config):

    num_iterations = 10
    superpixel_params = {
        'num_superpixels': config['R'],
        'num_levels': 4,
        'image_height': img.shape[0],
        'image_width': img.shape[1],
        'image_channels': img.shape[2]
    }

    seeds = cv2.ximgproc.createSuperpixelSEEDS(**superpixel_params)
    seeds.iterate(img, num_iterations)

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    superpixels_mean = mean_per_sp(img_lab, seeds)

    clustering = DBSCAN(eps=config['t_color'], metric=lab_distance, min_samples=2).fit(superpixels_mean)

    sp_neighbors = get_sp_neighboors(seeds)
    neighbors_cluster = get_neighbors_cluster(clustering, sp_neighbors)
    megapixel = merge_neighbors(img, seeds, neighbors_cluster)

    # mask_slic = seeds.getLabelContourMask()
    # mask_inv_slic = cv2.bitwise_not(mask_slic)
    # img_slic = cv2.bitwise_and(img, img, mask=mask_inv_slic)
    # cv2.imshow("img_seeds", img_slic)
    #
    # megapixel = np.uint8(megapixel)
    # img_megapixel = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(cv2.Canny(megapixel, 1, 1)))
    # cv2.imshow("img_megapixel", img_megapixel)
    # cv2.waitKey(0)

    return megapixel

def mean_per_sp(img, seeds):

    labels = seeds.getLabels()
    u_sp = []

    for sp_id in range(seeds.getNumberOfSuperpixels()):
        sp_coordinates = np.where(labels == sp_id)

        img_sp = [img[i][j] for i, j in zip(sp_coordinates[0], sp_coordinates[1])]
        u_sp.append(np.mean(img_sp, axis=0))

    return np.array(u_sp)

def merge_neighbors(img, seeds, neighbors_cluster):

    megapixel = np.zeros(img.shape[:2])
    labels = seeds.getLabels()

    for megapixel_id, neighbor_cluster in neighbors_cluster.items():
        for neighbor in neighbor_cluster:
            coordinates = np.where(labels == neighbor)
            megapixel[coordinates] = megapixel_id

    return megapixel

def lab_distance(p, q):

    return abs(p[0] - q[0]) + abs(p[1] - q[1]) + abs(p[2] - q[2])

def get_neighbors_cluster(clustering, neighbors):

    neighbors_cluster = {}

    for cluster_id in set(clustering.labels_):
        sp_candidates = list(np.where(cluster_id == clustering.labels_)[0])

        groups = []
        while len(sp_candidates) > 0:
            group, stack = [], []
            curr_candidate = sp_candidates.pop(0)

            stack.append(curr_candidate)

            while len(stack) > 0:
                elem = stack.pop(0)
                group.append(elem)

                for neighbor in neighbors[elem]:
                    if neighbor in sp_candidates:
                        stack.append(neighbor)
                        del sp_candidates[sp_candidates.index(neighbor)]

            groups.append(group)

        for i, group in enumerate(groups, start=len(neighbors_cluster)):
            neighbors_cluster[i] = group

    return neighbors_cluster

def get_sp_neighboors(seeds):

    labels = seeds.getLabels()

    h, w = labels.shape
    neighbors = {k: set() for k in range(seeds.getNumberOfSuperpixels())}

    for i, row in enumerate(labels):
        for j, p in enumerate(row):

            if j + 1 != w:
                if labels[i][j + 1] != p:
                    neighbors[p].add(labels[i][j + 1])
                    neighbors[labels[i][j + 1]].add(p)

            if i + 1 != h:
                if labels[i + 1][j] != p:
                    neighbors[p].add(labels[i + 1][j])
                    neighbors[labels[i + 1][j]].add(p)

    return neighbors

def motion_from_appearance(pm, img, config):
    x, y = np.indices((img.shape[0], img.shape[1]))
    features = np.array([[*img[i - 1][j - 1], *img[i - 1][j], *img[i][j + 1],
                          *img[i][j - 1],     *img[i][j],     *img[i][j + 1],
                          *img[i + 1][j - 1], *img[i + 1][j], *img[i + 1][j + 1]] for i, j in zip(x.flatten(), y.flatten()) if
                          (i != 0) and (i != img.shape[0] - 1) and (j != 0) and (j != img.shape[1] - 1)])

    x, y = np.where(pm > config['p_h'])
    fg_features = np.array([[*img[i - 1][j - 1], *img[i - 1][j], *img[i][j + 1], *img[i][j - 1], *img[i][j], *img[i][j + 1], *img[i + 1][j - 1], *img[i + 1][j], *img[i + 1][j + 1]] for i, j in zip(x, y) if (i != 0) and (i != img.shape[0] - 1) and (j != 0) and (j != img.shape[1] - 1)])

    x, y = np.where(pm < config['p_l'])
    bg_features = np.array([[*img[i - 1][j - 1], *img[i - 1][j], *img[i][j + 1], *img[i][j - 1], *img[i][j], *img[i][j + 1], *img[i + 1][j - 1], *img[i + 1][j], *img[i + 1][j + 1]] for i, j in zip(x, y) if (i != 0) and (i != img.shape[0] - 1) and (j != 0) and (j != img.shape[1] - 1)])

    if len(fg_features) < 20 or len(bg_features) < 20:
        return pm

    fg_gmm = GaussianMixture(n_components=7, tol=config['C']).fit(fg_features)
    bg_gmm = GaussianMixture(n_components=9, tol=config['C']).fit(bg_features)

    l_fg = np.divide(fg_gmm.score_samples(features), 9)
    l_bg = np.divide(bg_gmm.score_samples(features), 9)
    llr = np.divide(l_fg, l_bg)

    pa = 2 / (1 + np.exp(-2 * llr)) - 1

    pa_t = pm.copy()
    pa_t[1:-1, 1:-1] = pa.reshape(img.shape[0] - 2, img.shape[1] - 2)

    return pa_t

def motion_from_optical_flow(motion_vector, dynamic_model, config):

    motion_vector_, dynamic_model_ = motion_vector, dynamic_model
    epsilon, prev_residual = config['motion_detection']['epsilon'] + 1e-6, 0
    x, y = config['x'], config['y']

    while epsilon > config['motion_detection']['epsilon']:
        # Fit a dynamic model to actual motion
        coefficients, r, rank, s = np.linalg.lstsq(a=dynamic_model_, b=motion_vector_, rcond=None)
        epsilon = abs(r - prev_residual)
        prev_residual = r

        # Compute estimated motion, pixel wise motion error and foreground probability
        a, b, c, d, e, f = coefficients
        estimated_motion = np.array(a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y + f).T
        pixel_wise_motion_error = np.abs(np.subtract(motion_vector, estimated_motion))
        fg_probability = activation_function(pixel_wise_motion_error)

        # Select outlier pixels
        outlier_mask = fg_probability < config['motion_detection']['t_motion']
        outliers_index = np.where(outlier_mask == True)

        # Update models
        dynamic_model_ = dynamic_model[outliers_index]
        motion_vector_ = motion_vector[outliers_index]

    return fg_probability.reshape(config['height'], config['width'])

def motion_probability(img_1, img_2, optical_flow_model, dynamic_model, config):

    # Get motion vector
    motion_vector = get_motion_vector(optical_flow_model, img_1, img_2, config)
    motion_vector = get_pca_projections(motion_vector)

    # Run motion module
    pm = motion_from_optical_flow(motion_vector, dynamic_model, config)
    fg_mask = pm < config['motion_detection']['t_motion']

    return fg_mask

if __name__ == '__main__':

    # Load configurations
    parser = argparse.ArgumentParser(description='Motion detector')
    parser.add_argument('--config_path', type=str, help='Path to configuration file', default='configuration/motion_detection_params.yaml')

    # RAFT args
    parser.add_argument('--model', default='models/raft/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--image_path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficient correlation implementation')
    parser.add_argument('--debug', default=True, help='Activate / deactivate debug')
    args = parser.parse_args()

    debug = args.debug
    fps, cap, config, X, Y, dynamic_model, optical_flow_model = initialize(args)

    #visualize_optical_flow(optical_flow_model)
    while True:

        start = time.time()

        _, img_1 = cap.read(); img_1 = cv2.resize(img_1, (config['width'], config['height']), interpolation=cv2.INTER_AREA)
        _, img_2 = cap.read(); img_2 = cv2.resize(img_2, (config['width'], config['height']), interpolation=cv2.INTER_AREA)

        # Get motion vector
        motion_vector = get_motion_vector(optical_flow_model, img_1, img_2)
        motion_vector = get_pca_projections(motion_vector)

        # Run motion module
        pm = motion_from_optical_flow(motion_vector, dynamic_model, config)

        # Run appearance module
        # pa = motion_from_appearance(pm, img_2, config)
        #
        # Combine with megapixel
        #megapixel = get_mega_pixel(img_1, config)
        #for megapixel_id in range(len(megapixel)):
        #    mp_coordinates = np.where(megapixel == megapixel_id)
        #
        #    pm_avg = np.mean(pm[mp_coordinates])
        #    pm[mp_coordinates] = pm_avg
        #
        #     pa_avg = np.mean(pa[mp_coordinates])
        #     pa[mp_coordinates] = pa_avg

        # Visualize motion segmentation
        fg_mask = pm < config['t_motion']
        visualize_motion_segmentation(fg_mask, config, img_1)

        # Prints performance information
        elapsed_time = time.time() - start
        fps.append(1 / elapsed_time)
        if debug:
            print(f'FPS: {fps[-1]:.1f}\nFPS Avg: {np.mean(fps):.1f}\n')

        if cv2.waitKey(0) == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)
