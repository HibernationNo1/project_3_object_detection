

import mmcv
from mmcv import Config
import argparse
import os
import glob
import time
from tqdm import tqdm
import json
import numpy as np
import PIL
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw

from ast import parse
import hashlib


# python labelme.py test --cfg configs/labelme_config.py --ctgr strawberry


class labelme_custom():
    """
        
    """
    def __init__(self, cfg, annnotation_dir_path, dataset_dir_path):
        self.cfg = cfg
        self.dataset_dir_path = dataset_dir_path
        self.annnotation_dir_path = annnotation_dir_path
        
        self.dataset = dict(images = [], annotations = [], categories = [])
        self.object_names = []
        
        
        self.data_transfer()
    
    def data_transfer(self):
        labelme_json_list = glob.glob(os.path.join(self.annnotation_dir_path, "*.json"))
        
        self.get_images_info(labelme_json_list)
        
        self.get_annotations_info(labelme_json_list)

        # for i, json_file in enumerate(labelme_json_list):
        #     with open(json_file, "r") as fp:
        #         data = json.load(fp) 
        #         data['imageData']
           

    def get_annotations_info(self, labelme_json_list):
        for i, json_file in enumerate(labelme_json_list):
            with open(json_file, "r") as fp:
                data = json.load(fp) 

                image_height, image_width = data["imageHeight"], data["imageWidth"]
  
                for shape in data['shapes']:    # shape은 1개의 object.     1개의 image마다 1개 이상의 object가 있다.
                    tmp_annotations_dict = {}
                    if shape['shape_type'] == "polygon":     # segmentation labeling 
                        # tmp_annotations_dict['image_id'] =           
                        tmp_annotations_dict['segmentation'] = shape['points']
                        contour = np.array(tmp_annotations_dict['segmentation'])
                        mask = self.polygons_to_mask([image_height, image_width], contour)
                        tmp_annotations_dict['bbox'] = list(map(float, self.mask2box(mask)))
                        tmp_annotations_dict['image_id'] = i
                        
                        tmp_annotations_dict['iscrowd'] = 0     # TODO iscrowd ==  1인 경우의 dataset다룰때 사용해보기. 
                        
                        if shape['label'] not in self.object_names:
                            self.object_names.append(shape['label'])
                        
                        tmp_annotations_dict['object_id'] = self.object_names.index(shape['label'])         # object_id 와 category_id를 통일 
                        tmp_annotations_dict['category_id'] = self.object_names.index(shape['label'])
                        
                        self.dataset['annotations'].append(tmp_annotations_dict)
                        
                    else: continue      # TODO : point 형태의 annotation인 경우 
                        
    
        
    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))  
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask
    
    
    def mask2box(self, mask):
        index = np.argwhere(mask == 1)   
        
        rows = index[:, 0]
        clos = index[:, 1]       

        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        
        return [
            left_top_c,
            left_top_r,
            right_bottom_c - left_top_c,
            right_bottom_r - left_top_r,
        ]

    def get_images_info(self, labelme_json_list):
        for i, json_file in enumerate(labelme_json_list):
            tmp_images_dict = {}
            with open(json_file, "r") as fp:
                data = json.load(fp) 
                tmp_images_dict['file_name'] = data['imagePath']
                tmp_images_dict['width'] = data["imageWidth"]
                tmp_images_dict['height'] = data["imageHeight"]
                tmp_images_dict['image_id'] = i
                
            self.dataset["images"].append(tmp_images_dict)


def set_data_root(cfg, args):
    labelme_path = os.path.abspath(cfg.dir_info.labelme_dir)
    
    # checking dir path is exist
    annotations_path = os.path.join(labelme_path, cfg.dir_info.annotations_dir)
    annotations_category_path = os.path.join(annotations_path, args.ctgr)
    if not os.path.isdir(annotations_category_path):
        raise IOError(f' check directory path! : {annotations_category_path}')
    
    # set dir path to save dataset
    yyyy_mm_dd_hhmm = time.strftime('%Y-%m-%d', time.localtime(time.time())) \
                       + "-"+ str(time.localtime(time.time()).tm_hour) \
                       + str(time.localtime(time.time()).tm_min)
    dataset_path = os.path.join(labelme_path, cfg.dir_info.dataset_dir)
    dataset_category_path = os.path.join(dataset_path, yyyy_mm_dd_hhmm + "_" + args.ctgr)
    
    return annotations_category_path, dataset_category_path


def set_config(args):
    cfg = Config.fromfile(args.cfg)
    
    if args.json_name is not None: 
        if os.path.splitext(args.json_name)[1] !=".json":
            raise IOError('Only .json type are supoorted now!')
        cfg.json.file_name = args.json_name
    
    return cfg

def parse_args():
    parser = argparse.ArgumentParser(description="labelme annotation to custom data json file.")
    parser.add_argument('target_dir_name', help = "directory to labelme images and annotation json files.")
    parser.add_argument("--cfg", required = True, help="name of config file")
    parser.add_argument('--ctgr', required = True, help= "category of dataset")
    parser.add_argument('--json_name', help="name of json file")
    
    args=parser.parse_args()
    
    return args

if __name__ == "__main__":

    args = parse_args()
    
    cfg = set_config(args)
    
    annnotation_dir_path, dataset_dir_path = set_data_root(cfg, args)
    
    labelme_custom(cfg, annnotation_dir_path, dataset_dir_path)
    
        