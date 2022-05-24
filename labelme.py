

import mmcv
from mmcv import Config
import argparse
from ast import parse
import hashlib

# python labelme.py test --cfg configs/labelme_config.py --cgr strawberry

def parse_args():
    parser = argparse.ArgumentParser(description="labelme annotation to custom data json file.")
    parser.add_argument('target_dir_name', help = "directory to labelme images and annotation json files.")
    parser.add_argument("--cfg", required = True, help="name of config file")
    parser.add_argument('--cgr', required = True, help= "main plant category of dataset")
    parser.add_argument('--file_name', help="name of json file")
    
    args=parser.parse_args()
    
    return args

if __name__ == "__main__":

    args = parse_args()
    
    cfg = Config.fromfile(args.cfg)
    
    
        