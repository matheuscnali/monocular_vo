# Monocular Visual Odometry

Monocular visual odometry for autonomous navigation systems.

## Detectors
* FAST
* SuperPoint

## Trackers
* Optical Flow
* Nearest Neighbor

## Image Source
* Local image files
* Camera

## Ground truth source
* Local ground truth file (KITTI dataset)

## Roadmap
* Analyse current VO system performance in dynamic scenes.
* Apply [movement detection algorithm](https://www.sciencedirect.com/science/article/abs/pii/S0923596518309779) to improve VO.
* Investigate photometric calibration for optical flow.
* Compare detectors and trackers.
* Add outliers in scene image.
* Investigate methods for scale estimation.
* Investigate bundle adjustment.
* Implement camera image retification.
* Make image source params mutually exclusive.

## References
Some references that helped me to develop this project.
* [Chapel, M.N. and Bouwmans, T., 2020. Moving Objects Detection with a Moving Camera: A Comprehensive Review. arXiv preprint arXiv:2001.05238.](https://arxiv.org/abs/2001.05238)

* [SuperPoint-Visual Odometry (SP-VO)](https://github.com/syinari0123/SuperPoint-VO)

* [OpenCV documentation - Camera Calibration and 3D Reconstruction.](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html)

* [Fiani, S. and Fravolini, M.L., 2010. A robust monocular visual odometry algorithm for autonomous robot application. IFAC Proceedings Volumes, 43(16), pp.551-556.](https://www.sciencedirect.com/science/article/pii/S1474667016351151)

* [Forster, C., Lynen, S., Kneip, L. and Scaramuzza, D., 2013, November. Collaborative monocular slam with multiple micro aerial vehicles. In 2013 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 3962-3970). IEEE.](https://ieeexplore.ieee.org/abstract/document/6696923)

* [Visual Odometry (VO) - Presented by Patrick McGarey.](http://www.cs.toronto.edu/~urtasun/courses/CSC2541/03_odometry.pdf)

* [Geiger, A., Lenz, P. and Urtasun, R., 2012, June. Are we ready for autonomous driving? the kitti vision benchmark suite. In 2012 IEEE Conference on Computer Vision and Pattern Recognition (pp. 3354-3361). IEEE.](https://ieeexplore.ieee.org/abstract/document/6248074)

* [ZHANG, Ge, 2019, "Dynamic Scenes Dataset for Visual SLAM", https://doi.org/10.7910/DVN/NZETVT, Harvard Dataverse, V5](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/NZETVT)



# Developer
[Matheus Carvalho Nali](nalimatheus@gmail.com)

