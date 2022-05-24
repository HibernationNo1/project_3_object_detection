dir_info = dict(
    dataset_dir = "train_dataset",
    org_images_dir = 'dataset'
)

option = dict(
    save_gt_image = False,
    visul_gt_image_dir = 'gt_images',
    augmentation = dict(
        augmentation_flag = False,   # dataset자체적으로 augmentation진행할지 여부
        resize_flag = False,            # resizing으로 augmentation
        resizing_max_size = 1280,       # resizing을 수행할 때 긴 면의 size
        vertivcal_flip_flag = False     # flip으로 augmentation
    ),
    non_point = True        # point 라벨링이 되어있는 dataset이 포함되어 있지만, point label을 사용하지 않을 경우 True
)

json = dict(
    vaild_type=['paprika', "strawberry", "melon", 'onion', "seeding_pepper", 'cucumber', 'tomato'],
    vaild_object = ["leaf", 'midrid', 'stem', 'petiole', 'flower', 'fruit', 'y_fruit', 'cap', 
                    'first_midrid', 'last_midrid', 'mid_midrid', 'side_midrid'],
    file_name = 'dataset.json'
)