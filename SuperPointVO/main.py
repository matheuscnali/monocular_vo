import numpy as np
import cv2

from norm_visual_odometry import PinholeCamera, VisualOdometry
from sp_visual_odometry import VisualOdometry as sp_VisualOdometry

# kitti sequence
Num = "01"

# for each camera model
if Num in ["00", "01", "02"]:
    cam0_2 = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
elif Num in ["03"]:
    cam3 = PinholeCamera(1242.0, 375.0, 721.5377, 721.5377, 609.5593, 172.854)
elif Num in ["04", "05", "06", "07", "08", "09", "10"]:
    cam4_10 = PinholeCamera(1226.0, 370.0, 707.0912, 707.0912, 601.8873, 183.1104)
else:
    print("Error.")
    exit()

# pose_path
pose_path = '/media/az/420a7b37-c2b0-476d-b5a9-36622869cdb0/dataset/poses/' + Num + '.txt'
vo = VisualOdometry(cam0_2, pose_path)
sp_vo = sp_VisualOdometry(cam0_2, pose_path)

traj = np.zeros((600, 600, 3), dtype=np.uint8)

# log
log_fopen = open("results/kitti_" + Num + ".txt", mode='a')

# list
sp_errors = []
norm_errors = []
sp_feature_nums = []
norm_feature_nums = []

cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE) 
cv2.namedWindow("Trajectory", cv2.WINDOW_AUTOSIZE)

for img_id in range(4541):
    img = cv2.imread('/media/az/420a7b37-c2b0-476d-b5a9-36622869cdb0/dataset/sequences/' +
                     Num + '/image_0/' + str(img_id).zfill(6) + '.png', 0)

    # === superpoint ==============================
    sp_vo.update(img, img_id)

    sp_cur_t = sp_vo.cur_t
    if(img_id > 2):
        sp_x, sp_y, sp_z = sp_cur_t[0], sp_cur_t[1], sp_cur_t[2]
    else:
        sp_x, sp_y, sp_z = 0., 0., 0.

    sp_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for (u, v) in sp_vo.px_ref:
        cv2.circle(sp_img, (u, v), 3, (0, 255, 0))

    # === normal ==================================
    vo.update(img, img_id)

    cur_t = vo.cur_t
    if(img_id > 2):
        x, y, z = cur_t[0], cur_t[1], cur_t[2]
    else:
        x, y, z = 0., 0., 0.

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for (u, v) in vo.px_ref:
        cv2.circle(img, (u, v), 3, (0, 255, 0))

    # === calculation =============================
    # calculate error
    sp_est_point = np.array([sp_x, sp_z]).reshape(2)
    norm_est_point = np.array([x, z]).reshape(2)
    gt_point = np.array([sp_vo.trueX, sp_vo.trueZ]).reshape(2)
    sp_error = np.linalg.norm(sp_est_point - gt_point)
    norm_error = np.linalg.norm(norm_est_point - gt_point)

    # append
    sp_errors.append(sp_error)
    norm_errors.append(norm_error)
    sp_feature_nums.append(len(sp_vo.px_ref))
    norm_feature_nums.append(len(vo.px_ref))

    # average
    avg_sp_error = np.mean(np.array(sp_errors))
    avg_norm_error = np.mean(np.array(norm_errors))
    avg_sp_feature_num = np.mean(np.array(sp_feature_nums))
    avg_norm_feature_num = np.mean(np.array(norm_feature_nums))

    # === log writer ==============================
    print(img_id, len(sp_vo.px_ref), len(vo.px_ref),
          float(sp_x), float(sp_y), float(sp_z), float(x), float(y), float(z),
          sp_vo.trueX, sp_vo.trueY, sp_vo.trueZ, file=log_fopen)

    # === drawer ==================================
    # each point
    sp_draw_x, sp_draw_y = int(sp_x) + 290, int(sp_z) + 90
    norm_draw_x, norm_draw_y = int(x) + 290, int(z) + 90
    true_x, true_y = int(sp_vo.trueX) + 290, int(sp_vo.trueZ) + 90

    # draw trajectory
    cv2.circle(traj, (sp_draw_x, sp_draw_y), 1, (255, 0, 0), 1)
    cv2.circle(traj, (norm_draw_x, norm_draw_y), 1, (0, 255, 0), 1)
    cv2.circle(traj, (true_x, true_y), 1, (0, 0, 255), 2)
    cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)
    # draw text
    text = "Superpoint: [AvgFeature] %4.2f [AvgError] %2.4fm" % (
        avg_sp_feature_num, avg_sp_error)
    cv2.putText(traj, text, (20, 40),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, 8)
    text = "Normal    : [AvgFeature] %4.2f [AvgError] %2.4fm" % (
        avg_norm_feature_num, avg_norm_error)
    cv2.putText(traj, text, (20, 60),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, 8)

    cv2.imshow("Camera", np.concatenate((sp_img, img), axis=0))
    cv2.imshow("Trajectory", traj)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.imwrite('map.png', traj)
