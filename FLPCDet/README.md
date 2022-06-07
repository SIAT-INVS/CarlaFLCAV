## FLPCDet: Federated Learning Point CLoud Detection
Federated Learning Point CLoud Detection (FLPCDet) is based on Carla Simulation and OpenPCDet (See README_PCDET.md). 
FLPCDET can reproduce results in the following papers:

## Install FLPCDet
* FLPCDet is based on OpenPCDet (https://github.com/open-mmlab/OpenPCDet)
* OpenPCDet is based on SPCONV (https://github.com/traveller59/spconv)
* Test Environment: Ubuntu 20.04 (Python 3.8) and CUDA 11.3 (Nvidia Driver 470)

1. Install dependencies, e.g.,
```
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

2. Install OpenPCDet
```
cd FLPCDet
pip3 install -r requirements.txt
```

3. Install SPCONV
For CUDA 11.3, use
```
pip3 install spconv-cu113
```

4. Install OPEN3D
```
pip3 install open3d
```

5. Build 
```
python setup.py develop
```

## Quick Start

1. Prepare dataset
```
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/tesla339_dataset.yaml
```

2. Multi-stage Training
```
python sim_main.py
```

3.  Model Folders
* Fedmodel: Cloud and edge federated models
* Output: Local models at each autonomous vehicles

4. Testing
```
cd tools;
python test.py --cfg_file cfgs/kitti_models/town05_test/vehicle1232.yaml --ckpt ../fedmodel/cloud/global.pth
```

## Acknowledgement

* [CARLA](https://github.com/carla-simulator)
* [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)

## Citation

```tex
@article{FLCAV,
  title={Federated deep learning meets autonomous vehicle perception: Design and verification},
  author={Shuai Wang and Chengyang Li and Qi Hao and Chengzhong Xu and Derrick Wing Kwan Ng and Yonina C. Eldar and H. Vincent Poor},
  journal={arXiv preprint arXiv:2206.01748},
  year={2022}
}
```
