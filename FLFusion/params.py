#!/usr/bin/env python3
from pathlib import Path

CARLA_PATH = Path('~/carla/0.9.12').expanduser()
ROOT_PATH  = Path(__file__).parent
CARLA_AGENT_PYTHON_PATH = CARLA_PATH 

#for gen_data
LOG_PATH   = ROOT_PATH / 'log'
RAW_DATA_PATH  = ROOT_PATH / 'raw_data'
COOK_DATA_PATH = ROOT_PATH / 'dataset'
#https://carla.readthedocs.io/en/latest/core_map/#changing-the-map
TOWN_MAP = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05'][4]
RAW_DATA_START  = 60#frame
RAW_DATA_END    = -10#frame
RAW_DATA_FREQ   = 1#Hz
RAW_DATA_FREQ_ALT=1#Hz, for img_list generation
