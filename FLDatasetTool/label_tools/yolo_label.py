#!/usr/bin/python3
import glob
import os
import sys
from pathlib import Path
from multiprocessing import Pool as ProcessPool

import pandas as pd

sys.path.append(Path(__file__).resolve().parent.parent.as_posix())  # repo path
from param import RAW_DATA_PATH, DATASET_PATH
from label_tools.yolov5.yolov5_helper import *


def gather_yolo_data(record_name: str, vehicle_name: str, rgb_camera_name: str, semantic_camera_name: str):
    yolo_rawdata_df = pd.DataFrame()
    vehicle_rawdata_path = f"{RAW_DATA_PATH}/{record_name}/{vehicle_name}"
    rgb_image_path_list = sorted(glob.glob(f"{vehicle_rawdata_path}/{rgb_camera_name}/*.png"))
    semantic_image_path_list = sorted(glob.glob(f"{vehicle_rawdata_path}/{semantic_camera_name}/*.png"))
    yolo_rawdata_df['rgb_image_path'] = rgb_image_path_list
    yolo_rawdata_df['semantic_image_path'] = semantic_image_path_list
    yolo_rawdata_df['record_name'] = record_name
    yolo_rawdata_df['vehicle_name'] = vehicle_name
    return yolo_rawdata_df


class YoloLabelTool:
    def __init__(self):
        self.rec_pixels_min = 150
        self.color_pixels_min = 30
        self.debug = False

    def process(self, rawdata_df: pd.DataFrame):
        start = time.time()
        pool = ProcessPool()
        pool.starmap(self.process_frame, rawdata_df.iterrows())
        pool.close()
        pool.join()
        print("cost: {:0<3f}s".format(time.time() - start))

        # start = time.time()
        # for index, frame in rawdata_df.iterrows():
        #     self.process_frame(index, frame)
        # print("cost: {:0<3f}s".format(time.time() - start))

    def process_frame(self, index, frame):
        rgb_img_path = frame['rgb_image_path']
        seg_img_path = frame['semantic_image_path']
        success = check_id(rgb_img_path, seg_img_path)
        if not success:
            return

        output_dir = f"{DATASET_PATH}/{frame['record_name']}/{frame['vehicle_name']}/yolo"
        frame_id = get_filename_from_fullpath(rgb_img_path)

        image_rgb = None
        image_seg = None
        image_rgb = cv2.imread(rgb_img_path, cv2.IMREAD_COLOR)
        image_seg = cv2.imread(seg_img_path, cv2.IMREAD_UNCHANGED)
        if image_rgb is None or image_seg is None:
            return
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
        image_seg = cv2.cvtColor(image_seg, cv2.COLOR_BGRA2RGB)
        img_name = os.path.basename(rgb_img_path)
        height, width, _ = image_rgb.shape

        labels_all = []
        for index, label_info in LABEL_DATAFRAME.iterrows():
            seg_color = label_info['color']
            coco_id = label_info['coco_names_index']

            mask = (image_seg == seg_color)
            tmp_mask = (mask.sum(axis=2, dtype=np.uint8) == 3)
            mono_img = np.array(tmp_mask * 255, dtype=np.uint8)

            preview_img = image_rgb
            # self.preview_img = self.image_seg
            # cv2.imshow("seg", self.preview_img)
            # cv2.imshow("mono", mono_img)
            # cv2.waitKey()
            contours, _ = cv2.findContours(mono_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            labels = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h < YoloConfig.rectangle_pixels_min:
                    continue
                # cv2.rectangle(self.preview_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                # cv2.imshow("rect", self.preview_img)
                # cv2.waitKey()
                max_y, max_x, _ = image_rgb.shape
                if y + h >= max_y or x + w >= max_x:
                    continue

                if coco_id == TL_LIGHT_LABEL["DEFAULT"]:
                    coco_id = check_color(image_rgb[y:y + h, x:x + w, :])

                # DEBUG START
                # Draw label info to image
                # cv2.putText(preview_img, COCO_NAMES[coco_id], (x, y - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
                # cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 1)
                # DEBUG END

                label_info = "{} {} {} {} {}".format(coco_id,
                                                     float(x + (w / 2.0)) / width,
                                                     float(y + (h / 2.0)) / height,
                                                     float(w) / width,
                                                     float(h) / height)
                labels.append(label_info)
                # cv2.imshow("result", self.preview_img)
                # cv2.imshow("test", self.preview_img[y:y+h, x:x+w, :])
                # cv2.waitKey()

            if len(labels) > 0:
                labels_all += labels

        if len(labels_all) > 0:

            # print("Label output path: {}".format(self.label_out_path))
            write_image(output_dir, frame_id, image_rgb)
            write_label(output_dir, frame_id, labels_all)
        write_yaml(output_dir)
        return


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--record', '-r',
        required=True,
        help='Rawdata Record ID. e.g. record_2022_0113_1337'
    )
    argparser.add_argument(
        '--vehicle', '-v',
        default='all',
        help='Vehicle name. e.g. `vehicle.tesla.model3_1`. Default to all vehicles. '
    )
    argparser.add_argument(
        '--rgb_camera', '-c',
        default='image_2',
        help='Camera name. e.g. image_2'
    )
    argparser.add_argument(
        '--semantic_camera', '-s',
        default='image_2_semantic',
        help='Camera name. e.g. image_2_semantic'
    )

    args = argparser.parse_args()

    record_name = args.record
    if args.vehicle == 'all':
        vehicle_name_list = [os.path.basename(x) for x in
                             glob.glob('{}/{}/vehicle.*'.format(RAW_DATA_PATH, record_name))]
    else:
        vehicle_name_list = [args.vehicle]

    yolo_label_tool = YoloLabelTool()
    for vehicle_name in vehicle_name_list:
        rawdata_df = gather_yolo_data(args.record,
                                      vehicle_name,
                                      args.rgb_camera,
                                      args.semantic_camera)
        print("Process {} - {}".format(record_name, vehicle_name))
        yolo_label_tool.process(rawdata_df)


if __name__ == '__main__':
    main()

