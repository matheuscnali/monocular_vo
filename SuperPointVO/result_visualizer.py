import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_log(log_file):
    loaded_data = []
    img_ids = []
    sp_features = []
    norm_features = []
    sp_points = []
    norm_points = []
    gt_points = []
    with open(log_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            tmp_data = line.split()

            img_ids.append(int(tmp_data[0]))
            sp_features.append(int(tmp_data[1]))
            norm_features.append(int(tmp_data[2]))
            sp_points.append([float(x) for x in tmp_data[3:6]])
            norm_points.append([float(x) for x in tmp_data[6:9]])
            gt_points.append([float(x) for x in tmp_data[9:12]])
    return np.array(img_ids), np.array(sp_features), np.array(norm_features), np.array(sp_points), np.array(norm_points), np.array(gt_points)


def main():
    # dataloader
    img_ids, sp_features, norm_features, sp_points, norm_points, gt_points = read_log(
        "results/kitti_10.txt")
    # error
    sp_error = np.linalg.norm((sp_points - gt_points)[:, [0, 2]], axis=1)
    norm_error = np.linalg.norm((norm_points - gt_points)[:, [0, 2]], axis=1)
    # average error
    avg_sp_error = [np.mean(sp_error[:i]) for i in range(len(sp_error))]
    avg_norm_error = [np.mean(norm_error[:i]) for i in range(len(norm_error))]

    print("SuperPoint : ", avg_sp_error[-1])
    print("Normal     : ", avg_norm_error[-1])

    # visualize
    figure = plt.figure()
    if True:
        plt.subplot(2, 1, 1)
        plt.plot(img_ids, norm_features, color="blue", label="Normal-VO")
        plt.plot(img_ids, sp_features, color="red", label="SP-VO")
        plt.ylabel("Feature Number")

        plt.subplot(2, 1, 2)
        plt.plot(img_ids, avg_norm_error, color="blue", label="Normal-VO")
        plt.plot(img_ids, avg_sp_error, color="red", label="SP-VO")
        plt.xlabel("Timestamp")
        plt.ylabel("Avg Distance Error [m]")
        plt.legend()
    else:
        plt.plot(gt_points[:, 0], gt_points[:, 2],
                 color="black", label="Ground Truth")
        plt.plot(norm_points[:, 0], norm_points[:, 2],
                 color="blue", label="Normal-VO")
        plt.plot(sp_points[:, 0], sp_points[:, 2], color="red", label="SP-VO")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.axes().set_aspect('equal', 'datalim')
        plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
