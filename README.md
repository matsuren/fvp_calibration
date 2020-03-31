# FVP Calibration
Calibration for [Free Viewpoint Image Generation System using Fisheye Cameras and a Laser Rangefinder for Indoor Robot Teleoperation](http://dx.doi.org/10.1186/s40648-020-00163-4).

Please check [our project page](https://matsuren.github.io/fvp) for details.

If you use this code for your academic research, please cite the following paper.
```
@article{komatsu2020fvp,
  title={Free viewpoint image generation system using fisheye cameras and a laser rangefinder for indoor robot teleoperation},
  author={Komatsu, Ren and Fujii, Hiromitsu and Tamura, Yusuke and Yamashita, Atsushi and Asama, Hajime},
  journal={ROBOMECH Journal},
  volume={7},
  number={15},
  pages={1--10},
  year={2020},
  publisher={Springer}
}
```

## Environment
Visual Studio 2015, 2017, 2019

## Installation
Please use [vcpkg](https://github.com/microsoft/vcpkg). For x64-windows,
```
.\vcpkg install ceres[suitesparse,cxsparse]:x64-windows
```

Besides, OpenCV is also required. 

## Calibration
You need to calibrate intrinsic parameters of the cameras first using [OcamCalib](https://sites.google.com/site/scarabotix/ocamcalib-toolbox). Also, you need a 3D model for a robot. We used [colmap](https://colmap.github.io/) to create the robot model from multiple images.



Further information will be available soon!
