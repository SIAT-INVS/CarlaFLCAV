#! /usr/bin/python3
import argparse
import csv
import glob
import os
import pickle
import sys
import time
import pandas as pd
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())
from param import RAW_DATA_PATH, DATASET_PATH


def get_frame_from_fullpath(path: str) -> int:
    return int(os.path.splitext(os.path.split(path)[-1])[0])


def load_object_labels(path: str):
    object_labels_path_list = sorted(glob.glob("{}/*.pkl".format(path)))
    object_labels_df = pd.DataFrame(columns=['frame', 'object_labels_path'])
    for objects_labels_rawdata_path in object_labels_path_list:
        frame = get_frame_from_fullpath(objects_labels_rawdata_path)
        object_labels_df = object_labels_df.append({'frame': frame,
                                                    'object_labels_path': objects_labels_rawdata_path},
                                                   ignore_index=True)
    return object_labels_df


def gather_rawdata_to_dataframe(record_name: str):
    rawdata_frames_df = pd.DataFrame()
    object_labels_path_df = load_object_labels("{}/{}/others.world_0".format(RAW_DATA_PATH, record_name))
    rawdata_frames_df = object_labels_path_df
    rawdata_frames_df = rawdata_frames_df.reset_index(drop=False)
    return rawdata_frames_df


class ArgoverseLabelTool:
    def __init__(self, record_name, rawdata_df: pd.DataFrame, output_dir=None):
        self.record_name = record_name
        self.rawdata_df = rawdata_df
        self.output_dir = output_dir
        self.fieldnames = ['TIMESTAMP',
                           'TRACK_ID',
                           'OBJECT_TYPE',
                           'X',
                           'Y',
                           'CITY_NAME']
        self.is_first_frame = True
        self.av_id = -1
        self.agent_id = -1

    def process(self):
        start = time.time()

        # Output dataset in kitti format
        if self.output_dir == '':
            self.output_dir = f"{DATASET_PATH}/{self.record_name}/argoverse"
        else:
            self.output_dir = f"{DATASET_PATH}/{self.output_dir}/argoverse"

        os.makedirs(self.output_dir, exist_ok=True)

        with open('{}/output.csv'.format(self.output_dir), 'w', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            writer.writeheader()

        for index, frame in self.rawdata_df.iterrows():
            self.process_frame(index, frame)

        print("Cost: {:0<3f}s".format(time.time() - start))

    def process_frame(self, index, frame):
        index = "{:0>6d}".format(index)
        frame_id = "{:0>6d}".format(frame['frame'])

        # Load object labels from pickle data
        with open(frame['object_labels_path'], 'rb') as pkl_file:
            objects_labels = pickle.load(pkl_file)

        if self.is_first_frame:
            # Show all objects id
            id_list = []
            for label in objects_labels:
                if label.carla_id > 999999999999:
                    continue
                id_list.append(label.carla_id)
            sorted(id_list)
            print(id_list)

            # Select agent ID and AV ID
            self.av_id = int(input("Please input AV ID:"))
            self.agent_id = int(input("Please input Agent ID:"))
            self.is_first_frame = False

        for label in objects_labels:
            timestamp = label.timestamp

            # Drop env static vehicles
            if label.carla_id > 999999999999:
                continue

            track_id = "00000000-0000-0000-0000-{:0>12d}".format(label.carla_id)

            if label.carla_id == self.av_id:
                obj_type = 'AV'
            elif label.carla_id == self.agent_id:
                obj_type = 'AGENT'
            else:
                obj_type = 'OTHERS'

            with open('{}/output.csv'.format(self.output_dir), 'a', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
                csv_line = {'TIMESTAMP': timestamp,
                            'TRACK_ID': track_id,
                            'OBJECT_TYPE': obj_type,
                            'X': label.transform.location.x,
                            'Y': label.transform.location.y,
                            'CITY_NAME': "MIA"
                            }
                writer.writerow(csv_line)


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--record', '-r',
        required=True,
        help='Rawdata Record ID. e.g. record_2022_0113_1337'
    )
    argparser.add_argument(
        '--output_dir', '-o',
        default='',
        help='Output dir in dataset folder'
    )

    args = argparser.parse_args()

    record_name = args.record

    rawdata_df = gather_rawdata_to_dataframe(args.record)
    print("Process {} ".format(record_name))
    argo = ArgoverseLabelTool(record_name, rawdata_df, args.output_dir)
    argo.process()


if __name__ == '__main__':
    main()
