

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
import cv2
import random

from ast import parse
import hashlib


# python labelme.py test --cfg configs/labelme_config.py --ctgr paprika

# TODO : logger추가

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
        

class labelme_custom():
    """

    """
    def __init__(self, cfg, annnotation_dir_path, dataset_dir_path):
        self.cfg = cfg
        self.dataset_dir_path = dataset_dir_path
        self.annnotation_dir_path = annnotation_dir_path
        
        # TODO : key = info는 추후 내용 채우기
        self.dataset = dict(info = dict(description = ' ', url = ' ', version = ' ', year = ' ', contributor = ' ', deta_created = ' '), 
                            licenses = [],
                            images = [], annotations = [], categories = [])
        self.object_names = []
        
        
        self.data_transfer()
        
        # json_file = self.dataset
        # print(f"json_file['info'].keys() : {json_file['info'].keys()} \n")
        # print(f"json_file['images'][0].keys() : {json_file['images'][0].keys()} \n")
        # print(f"json_file['categories'][0].keys() : {json_file['categories'][0].keys()} \n")
        # print(f"json_file['annotations'][0].keys() : {json_file['annotations'][0].keys()} \n")
        # print(f"json_file['annotations'][0]['image_id'] : {json_file['annotations'][0]['image_id']}, {type(json_file['annotations'][0]['image_id'])}")
        # exit()
        self.save_dataset()
        
        
        
        # TODO : 필요시 GT image확인하는 function만들기

    def save_dataset(self):
        self.make_dir()
        self.save_images()
        self.save_json()
        

    def make_dir(self):
        labelme_dir = os.path.abspath(cfg.dir_info.labelme_dir)
        os.makedirs(labelme_dir, exist_ok = True)
        dataset_dir = os.path.join(labelme_dir, cfg.dir_info.dataset_dir)
        os.makedirs(dataset_dir, exist_ok = True)
        os.makedirs(self.dataset_dir_path, exist_ok= True)
        self.org_images_dir = os.path.join(self.dataset_dir_path, self.cfg.dir_info.org_images_dir)
        os.makedirs(self.org_images_dir, exist_ok = False)
        

    def save_images(self):
        print(f"\n part : saving...")
        for image_dict in tqdm(self.dataset['images']):
            image_path = os.path.join(self.annnotation_dir_path, image_dict['file_name'])
            
            img = cv2.imread(image_path)
            img_save_path = os.path.join(self.org_images_dir, os.path.basename(image_dict['file_name']))
            cv2.imwrite(img_save_path, img)


    def save_json(self):
        print(f"Dataset name to save is : {self.cfg.json.file_name} \n")
        save_path = os.path.join(self.dataset_dir_path, self.cfg.json.file_name)
        json.dump(self.dataset, open(save_path, "w"), indent=4, cls=NpEncoder)
        
        print("Done!")
    
    
    def data_transfer(self):
        labelme_json_list = glob.glob(os.path.join(self.annnotation_dir_path, "*.json"))
        
        self.get_images_info(labelme_json_list)
        
        self.get_annotations_info(labelme_json_list)
        
        self.get_categories_info()   
    
    
    def get_categories_info(self):
        print(f"\n part : categories")
        for i, object_name in enumerate(tqdm(self.object_names)):
            tmp_categories_dict = {}
            tmp_categories_dict['supercategory'] = object_name
            tmp_categories_dict['id'] =self.object_names.index(object_name)
            tmp_categories_dict['name'] = object_name
            self.dataset['categories'].append(tmp_categories_dict)
        
           
    def get_annotations_info(self, labelme_json_list):
        print(f"\n part : annotations")
        for i, json_file in enumerate(tqdm(labelme_json_list)):
            with open(json_file, "r") as fp:
                data = json.load(fp) 

                image_height, image_width = data["imageHeight"], data["imageWidth"]
  
                for shape in data['shapes']:    # shape은 1개의 object.     1개의 image마다 1개 이상의 object가 있다.
                    if shape['label'] not in self.object_names:
                        if shape['label'] in self.cfg.json.valid_object:
                            self.object_names.append(shape['label'])
                        else: 
                            if self.cfg.options.only_val_obj: raise KeyError(f"{shape['label']} is not valid object.")   
                            else: continue


                    tmp_annotations_dict = {}
                    if shape['shape_type'] == "polygon":  
                                                
                        contour = np.array(shape['points'])
                        tmp_segmentation = []
                        points = list(np.asarray(contour).flatten())
                        for point in points:
                            tmp_segmentation.append(round(point, 2))
                        tmp_annotations_dict['segmentation'] = [tmp_segmentation]
                        
                        mask = self.polygons_to_mask([image_height, image_width], contour)
                        x = contour[:, 0]
                        y = contour[:, 1]
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                        tmp_annotations_dict["area"] = float(area)
                        
                        tmp_annotations_dict['iscrowd'] = 0     # TODO iscrowd ==  1인 경우의 dataset다룰때 사용해보기.
                        tmp_annotations_dict['image_id'] = i+1
                        tmp_annotations_dict['bbox'] = list(map(float, self.mask2box(mask)))
                        tmp_annotations_dict['category_id'] = self.object_names.index(shape['label'])
                        tmp_annotations_dict['id'] =  random.randrange(1, len(self.object_names) + 100)
                        
                        
                    else : pass     # TODO : segmentation이 아닌 dataset을 다룰 때 기능 추가

                    self.dataset['annotations'].append(tmp_annotations_dict)


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
        print(f"part : images")
        for i, json_file in enumerate(tqdm(labelme_json_list)):
            tmp_images_dict = {}
            with open(json_file, "r") as fp:
                data = json.load(fp) 
                
                tmp_images_dict['license'] = None
                tmp_images_dict['file_name'] = data['imagePath']
                tmp_images_dict['coco_url'] = None
                tmp_images_dict['height'] = data["imageHeight"]
                tmp_images_dict['width'] = data["imageWidth"]
                tmp_images_dict['date_captured'] = None
                tmp_images_dict['flickr_url'] = None
                tmp_images_dict['id'] = i+1
                
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
        
    if args.ctgr not in cfg.json.valid_categorys:
        raise KeyError(f"{args.ctgr} is not valid category.")
    else: cfg.json.category = args.ctgr
    
    return cfg

def parse_args():
    parser = argparse.ArgumentParser(description="labelme annotation to custom data json file.")
    parser.add_argument('target_dir_name', help = "directory to labelme images and annotation json files.")
    parser.add_argument("--cfg", required = True, help="name of config file")
    parser.add_argument('--ctgr', required = True, help= "category of dataset")
    parser.add_argument('--json_name', help="name of json file")
    
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":

    args = parse_args()
    
    cfg = set_config(args)
    
    annnotation_dir_path, dataset_dir_path = set_data_root(cfg, args)
    
    labelme_custom(cfg, annnotation_dir_path, dataset_dir_path)
    
        