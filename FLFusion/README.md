## FLFusion: Federated Learning With Vehicle-Road Fusion 
Federated Learning Vehicle-to-infrastructure Fusion (FLFusion) is based on Carla Simulation and CARLA_INVS (See https://github.com/zijianzhang/CARLA_INVS). 
FLFUSION can reproduce results in the following papers:

## Quick Start

1. Open CARLA
Start the CARLA server: (e.g., carla is installed in folder ~/carla/0.9.12)
```
cd ~/carla/0.9.12
./CarlaUE4.sh
```

2. Distributed Dataset Generation
Run the following command to generate the raw data:
```
python3 road/Scenario_Infrastructure.py record 261,262,263,254,255,256,193,194,195,196,264,265,266,267,268,269,80,170,167,116,110,94,166,76,88 158,226,247,191 4,7
```
Format: python3 road/Scenario_Infrastructure.py record NPC_vehicles autonomous_vehicles traffic_lights_IDs
NOTE that the raw data at the road sensor is stored into "road.sensor.lidar_xxx". (this folder name can be revised)

3. Dataset Calibration and Annotation
Run the following command to generate the KITTI dataset:
```
python3 road/Process.py raw_data/recordxxx
```
Format: python3 road/Process.py data_folder

4. Vehicle and Road Federated Distillation
Run the following command to fuse the maps from the road sensors and the vehicle sensors:
```
python3 road/V2I_fusion.py ./dataset/recordxxx img_list label00
```
* The generated labels are then used for multi-stage training in FLPCDet.
Format: python3 road/V2I_fusion.py data_folder frame_list label_folder(detection_results, ground truth...)


## Acknowledgement

* [CARLA Ros Bridge](https://github.com/carla-simulator/ros-bridge)
* [CARLA_INVS](https://github.com/zijianzhang/CARLA_INVS)

## Citation

```latex
@article{FLCAV,
  title={Federated deep learning meets autonomous vehicle perception: Design and verification},
  author={Shuai Wang and Chengyang Li and Qi Hao and Chengzhong Xu and Derrick Wing Kwan Ng and Yonina C. Eldar and H. Vincent Poor},
  journal={arXiv:2206.01748},
  year={2022},
  publisher={IEEE}
}
```
