# CarlaFLCAV

![carla_flcav](https://user-images.githubusercontent.com/15060244/171803004-f2f699d5-1a18-48b5-ac12-672a045ba837.png)


**CarlaFLCAV** is an open-source FLCAV simulation platform based on CARLA simulator that supports: 

* **Multi-modal dataset generation**: Including point-cloud, image, radar data with associated calibration, synchronization, and annotation

* **Training and inference**: Examples for CAV perception, including object detection, traffic sign detection, and weather classification

* **Various FL frameworks**: FedAvg, device selection, noisy aggregation, parameter selection, distillation, and personalization

* **Optimization based modules**: Network resource and road sensor pose optimization.

# Demo

![fusion](https://user-images.githubusercontent.com/38368612/206235843-e6b37409-0be7-460b-a94a-0694eba2e6fc.gif)

**Cooperative perceptioin with road sensors for federated distillation** 

# Test Environment

- Ubuntu 20.04
- Python 3.8
- CARLA 0.9.13
- CUDA 11.3 (Nvidia Driver 470)
- Pytorch 1.10.0

# Citation

CarlaFLCAV can reproduce results in the following papers:

```tex
@article{FLCAV,
  title={Federated deep learning meets autonomous vehicle perception: Design and verification},
  author={Shuai Wang and Chengyang Li and Derrick Wing Kwan Ng and Yonina C. Eldar and H. Vincent Poor and Qi Hao and Chengzhong Xu},
  journal={IEEE Network},
  year={2022}
}
```

Arxiv version: http://arxiv.org/abs/2206.01748


### Authors

[Shuai Wang](https://github.com/bearswang)

[Chengyang Li](https://github.com/KevinLADLee)


