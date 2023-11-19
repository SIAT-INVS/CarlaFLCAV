## FLYolo: Federated Learning You Look Only Once (FLYolo)
Federated Learning You Look Only Once (FLYolo) is based on Carla Simulation and Yolov5 (See yolov5/README.md). 

## Install FLYOLO
```
pip3 install opencv-python
pip3 install -r yolov5/requirements.txt
pip install cvxpy
```

## Example RawData Folder Structure 

```
rawdata
├── pretrain
├── town03
├── town05
└── test
    └── vehicle.tesla.model3_173
        ├── yolo_coco_carla.yaml
        └── yolo_dataset
            ├── images
            │   └── train
            │       ├── 0000000174.jpg
            │       ├── **********.txt
            │       └── 0000000263.jpg
            └── labels
                └── train
                    ├── 0000000174.txt
                    ├── **********.txt
                    └── 0000000281.txt

```

example rawdata: [FLYolo.tar.gz](https://hkustgz-my.sharepoint.com/:f:/g/personal/cli386_connect_hkust-gz_edu_cn/Em2lXlioF8dEjxLiAjH20bQBZwZaDTgfYT4QwHpD4sv5YA?e=vaNCl6)

## Quick Test for YOLOV5

1. Train Yolov5
```
python3 yolov5/train.py --img 640 --batch 8 --epochs 5 --data raw_data/pretrain/vehicle.tesla.model3_135/yolo_coco_carla.yaml --cfg yolov5/models/yolov5s.yaml  --weights yolov5s.pt
```

2. Test Result
Test dataset = test/vehicle.tesla.model3_173
```
python3 yolov5/detect.py --source 'raw_data/test/vehicle.tesla.model3_173/yolo_dataset/images/train/*.jpg' --weights yolov5s.pt 

```

3. Evaluate Result
```
python3 yolov5/val.py --data raw_data/test/vehicle.tesla.model3_173/yolo_coco_carla.yaml --weights yolov5s.pt 
```

## Federated learning under resource constraints for YOLOV5
```
python3 flcav_yolo.py -w 4096 -l 4096 --batch_size 4 --epoch 10 --pretrain_model yolov5s.pt
```

## Acknowledgement

* [CARLA](https://github.com/carla-simulator)
* [Yolov5](https://github.com/ultralytics/yolov5)

## Citation

```tex
@article{FLCAV,
  title={Federated deep learning meets autonomous vehicle perception: Design and verification},
  author={Shuai Wang and Chengyang Li and Derrick Wing Kwan Ng and Yonina C. Eldar and H. Vincent Poor and Qi Hao and Chengzhong Xu},
  journal={IEEE Network},
  year={2022}
}
```
