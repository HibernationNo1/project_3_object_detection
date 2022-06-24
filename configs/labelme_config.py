from __future__ import annotations
import time                      

mode = "labelme"

dir_info = dict(
    labelme_dir = "labelme",
    annotations_dir = "annotations",
    dataset_dir = "train_dataset",      
    org_images_dir = 'org_images'
)

options = dict(
    save_gt_image = False,
    visul_gt_image_dir = 'gt_images',
    augmentation = dict(
        augmentation_flag = False,   # dataset자체적으로 augmentation진행할지 여부
        resize_flag = False,            # resizing으로 augmentation
        resizing_max_size = 1280,       # resizing을 수행할 때 긴 면의 size
        vertivcal_flip_flag = False     # flip으로 augmentation
    ),
    only_val_obj = False        # valid_objec에 포함되지 않는 라벨이 있을 때 무시하는 경우 False, Error 발생시키는 경우 True
)

json = dict(
    category = None,
    valid_categorys=['paprika', "strawberry", "melon", 'onion', "seeding_pepper", 'cucumber', 'tomato', 'test'],
    valid_object = ["leaf", 'midrid', 'stem', 'petiole', 'flower', 'fruit', 'y_fruit', 'cap', 
                    'first_midrid', 'last_midrid', 'mid_midrid', 'side_midrid'],
    file_name = 'dataset.json'
    )

dataset = dict(
    info = dict(description = 'Hibernation Custom Dataset',
                url = ' ',
                version = '0.0.1',
                year = 2022,
                contributor = ' ',
                data_created = f"{time.strftime('%Y/%m/%d', time.localtime(time.time()))}"),
    licenses = dict(url = ' ',
                    id = 1,
                    name = ' ')   
    
)