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
    def __init__(self, cfg, args):
        self.args = args
        self.cfg = cfg
        
        self.dataset = dict(info = {}, 
                            licenses = [],
                            images = [], annotations = [], categories = [])
        self.object_names = []
        
        self.dataset_dir_path = None
        self.annnotation_dir_path = None
        
        self.set_data_root()

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
        
        
    def set_data_root(self):
        labelme_path = os.path.abspath(self.cfg.dir_info.labelme_dir)
        
        # checking dir path is exist
        annotations_path = os.path.join(labelme_path, self.cfg.dir_info.annotations_dir)
        annotations_category_path = os.path.join(annotations_path, self.args.ann)
        if not os.path.isdir(annotations_category_path):
            raise IOError(f' check directory path! : {annotations_category_path}')
        
        # set dir path to save dataset
        self.yyyy_mm_dd_hhmm = time.strftime('%Y-%m-%d', time.localtime(time.time())) \
                        + "-"+ str(time.localtime(time.time()).tm_hour) \
                        + str(time.localtime(time.time()).tm_min)
        dataset_path = os.path.join(labelme_path, self.cfg.dir_info.dataset_dir)
        self.deta_ann = self.yyyy_mm_dd_hhmm + "_" + self.args.ann
        dataset_category_path = os.path.join(dataset_path, self.deta_ann)
        
        
        self.dataset_dir_path = dataset_category_path
        self.annnotation_dir_path = annotations_category_path
        

    def make_dir(self):
        labelme_dir = os.path.abspath(self.cfg.dir_info.labelme_dir)
        os.makedirs(labelme_dir, exist_ok = True)
        dataset_dir = os.path.join(labelme_dir, self.cfg.dir_info.dataset_dir)
        os.makedirs(dataset_dir, exist_ok = True)
        os.makedirs(self.dataset_dir_path, exist_ok= True)
        self.org_images_dir = os.path.join(self.dataset_dir_path, self.cfg.dir_info.org_images_dir)
        os.makedirs(self.org_images_dir, exist_ok = False)
        

    def save_images(self):
        print(f"\n part_6 : saving...")
        for image_dict in tqdm(self.dataset['images']):
            image_path = os.path.join(self.annnotation_dir_path, image_dict['file_name'])
            
            img = cv2.imread(image_path)
            img_save_path = os.path.join(self.org_images_dir, os.path.basename(image_dict['file_name']))
            cv2.imwrite(img_save_path, img)


    def save_json(self):
        print(f"\n Dataset name to save is : {self.cfg.json.file_name} ")
        save_dataset_path = os.path.join(self.dataset_dir_path, self.cfg.json.file_name)
        json.dump(self.dataset, open(save_dataset_path, "w"), indent=4, cls=NpEncoder)                      # save dataset
        
        save_classes_path = os.path.join(self.dataset_dir_path, self.deta_ann + "_classes.json")
        json.dump(self.object_names, open(save_classes_path, "w"), indent=4, cls=NpEncoder)     # save classes    
        
        print("Done!")
    
    
    def data_transfer(self):
        labelme_json_list = glob.glob(os.path.join(self.annnotation_dir_path, "*.json"))
        
        self.get_info()
        self.get_licenses()
        self.get_images(labelme_json_list)
        self.get_annotations(labelme_json_list)
        self.get_categories()   
        

    def get_info(self) : 
        print(f" part_1 : info")
        self.dataset['info']['description'] = self.cfg.dataset.info.description
        self.dataset['info']['url']         = self.cfg.dataset.info.url
        self.dataset['info']['version']     = self.cfg.dataset.info.version
        self.dataset['info']['year']        = self.cfg.dataset.info.year
        self.dataset['info']['contributor'] = self.cfg.dataset.info.contributor
        self.dataset['info']['data_created']= self.cfg.dataset.info.data_created

    def get_licenses(self):
        print(f"\n part_2 : licenses")
        if self.cfg.dataset.licenses is not None:
            tmp_dict = dict(url = self.cfg.dataset.licenses.url,
                            id = self.cfg.dataset.licenses.id,
                            name = self.cfg.dataset.licenses.name)   
            self.dataset['licenses'].append(tmp_dict)  # 기존 coco dataset은 license가 여러개 존재
        else: 
            pass  
    
    def get_categories(self):
        print(f"\n part_5 : categories")
        for i, object_name in enumerate(tqdm(self.object_names)):
            tmp_categories_dict = {}
            tmp_categories_dict['supercategory'] = object_name                          # str
            tmp_categories_dict['id'] = self.object_names.index(object_name)            # int
            tmp_categories_dict['name'] = object_name                                    # str
            self.dataset['categories'].append(tmp_categories_dict)
        
           
    def get_annotations(self, labelme_json_list):
        print(f"\n part_4 : annotations")
        id_count = 1
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
                        
                        tmp_annotations_dict['iscrowd'] = None     # TODO iscrowd ==  1인 경우의 dataset다룰때 사용해보기.
                        tmp_annotations_dict['image_id'] = i+1
                        tmp_annotations_dict['bbox'] = list(map(float, self.mask2box(mask)))
                        tmp_annotations_dict['category_id'] = self.object_names.index(shape['label'])       # int, same to category_id
                        tmp_annotations_dict['id'] = id_count
                        id_count +=1
                        
                        
                    else : continue     # TODO : segmentation이 아닌 dataset을 다룰 때 기능 추가
 
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

    def get_images(self, labelme_json_list):
        print(f"\n part_3 : images")
        yyyy_mm_dd_hh_mm = time.strftime('%Y-%m-%d', time.localtime(time.time())) \
                       + " "+ str(time.localtime(time.time()).tm_hour) \
                       + ":" + str(time.localtime(time.time()).tm_min)
                       
        for i, json_file in enumerate(tqdm(labelme_json_list)):
            tmp_images_dict = {}
            with open(json_file, "r") as fp:
                data = json.load(fp) 
                
                tmp_images_dict['license'] = len(self.dataset['licenses'])  # license가 1개 임의의 값이기 때문에 1로 통일
                tmp_images_dict['file_name'] = data['imagePath']
                tmp_images_dict['coco_url'] = " "                       # str
                tmp_images_dict['height'] = data["imageHeight"]         # int
                tmp_images_dict['width'] = data["imageWidth"]           # int
                tmp_images_dict['date_captured'] = yyyy_mm_dd_hh_mm     # str   
                tmp_images_dict['flickr_url'] = " "                     # str
                tmp_images_dict['id'] = i+1                                 # 중복되지 않는 임의의 int값
                
            self.dataset["images"].append(tmp_images_dict)



