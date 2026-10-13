<div align="center">

<h1>CarlaFLCAV</h1>
<p><strong>Federated Learning for Connected Autonomous Vehicle Perception</strong></p>
<p>An open-source CARLA platform for multi-modal data collection and federated perception.</p>

<p>
  <a href="https://siat-invs.github.io/CarlaFLCAV/"><img src="https://img.shields.io/badge/Website-CarlaFLCAV-3c5488?style=for-the-badge" alt="Project Website" height="28"></a>
  <a href="https://arxiv.org/abs/2206.01748"><img src="https://img.shields.io/badge/arXiv-2206.01748-b31b1b?style=for-the-badge" alt="arXiv:2206.01748" height="28"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Cite-CarlaFLCAV-2ea44f?style=for-the-badge" alt="Cite CarlaFLCAV" height="28"></a>
  <a href="https://github.com/KevinLADLee/carla_dataset_tools"><img src="https://img.shields.io/badge/Data-Recorder-3c5488?style=for-the-badge" alt="Data Recorder" height="28"></a>
  <a href="https://github.com/SIAT-INVS/CarlaFLCAV/stargazers"><img src="docs/assets/combined-stars.svg" alt="GitHub stars" height="28"></a>
</p>

<p><a href="#overview">Overview</a> · <a href="#demo">Demos</a> · <a href="#test-environment">Environment</a> · <a href="#citation">Citation</a></p>

</div>

![carla_flcav](https://user-images.githubusercontent.com/15060244/171803004-f2f699d5-1a18-48b5-ac12-672a045ba837.png)


## Overview

**CarlaFLCAV** is an open-source FLCAV simulation platform based on CARLA simulator that supports: 

* **Multi-modal dataset generation**: Including point-cloud, image, radar data with associated calibration, synchronization, and annotation

* **Training and inference**: Examples for CAV perception, including object detection, traffic sign detection, and weather classification

* **Various FL frameworks**: FedAvg, device selection, noisy aggregation, parameter selection, distillation, and personalization

* **Optimization based modules**: Network resource and road sensor pose optimization.

The Data Recorder is based on our [carla_dataset_tools](https://github.com/KevinLADLee/carla_dataset_tools) repository.

## Demo

https://user-images.githubusercontent.com/38368612/206344921-f52956af-86bc-48b6-aee1-388febc233d7.mp4

**Federated SECOND for 3D point cloud object detection** 

https://user-images.githubusercontent.com/38368612/206344904-d2d7d194-d104-4701-a771-a97a6136e3a6.mp4

**Federated YOLOV5 for 2D image object detection** 

https://user-images.githubusercontent.com/38368612/206665655-c8653bb0-3e25-4071-9797-8e76255b4eab.mp4

**Federated LSTM for BEV trajectory prediction** 

https://user-images.githubusercontent.com/38368612/206345017-5c8a764a-44fb-4282-b832-cb8e55090d7d.mp4

**Cooperative perceptioin with road sensors for federated distillation** 



## Test Environment

- Ubuntu 20.04
- Python 3.8
- CARLA 0.9.13
- CUDA 11.3 (Nvidia Driver 470)
- Pytorch 1.10.0

## Citation

CarlaFLCAV can reproduce results in the following papers:

```tex
@article{CarlaFLCAV,
  title={Federated deep learning meets autonomous vehicle perception: Design and verification},
  author={Shuai Wang and Chengyang Li and Derrick Wing Kwan Ng and Yonina C. Eldar and H. Vincent Poor and Qi Hao and Chengzhong Xu},
  journal={IEEE Network},
  year={2023},
  volume={37},
  number={3},
  pages={16--25}
}

@article{CarlaFLOTA,
  title={Edge federated learning via unit-modulus over-the-air computation},
  author={Shuai Wang and Yuncong Hong and Rui Wang and Qi Hao and Yik-Chung Wu and Derrick Wing Kwan Ng},
  journal={IEEE Transactions on Communications},
  year={2022},
  volume={70},
  number={5},
  pages={3141--3156}
}
```

CarlaFLCAV Arxiv version: http://arxiv.org/abs/2206.01748

CarlaFLOTA Arxiv version: https://arxiv.org/abs/2101.12051

## Acknowledgement

* [Carla](https://github.com/carla-simulator)
* [Carla-ROS-Bridge](https://github.com/carla-simulator/ros-bridge)
* [CarlaINVS](https://github.com/zijianzhang/CARLA_INVS)

### Authors

[Shuai Wang](https://github.com/bearswang)

[Chengyang Li](https://github.com/KevinLADLee)


