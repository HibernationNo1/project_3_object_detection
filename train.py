

import os
from posixpath import split
import time, json
import warnings

from labelme import NpEncoder

import mmcv

from mmcv.utils import get_logger
from mmcv.utils import get_git_hash

import torch
import torch.distributed as dist

from custom_mmdet import __version__
from custom_mmdet.utils import collect_env, get_device
from custom_mmdet.apis import init_random_seed, set_random_seed, train_detector
from custom_mmdet.models import build_detector
from custom_mmdet.datasets import build_dataset



# from mmdet import __version__
# from mmdet.utils import collect_env, get_device
# from mmdet.apis import init_random_seed, set_random_seed, train_detector
# from mmdet.models import build_detector
# from mmdet.datasets import build_dataset

def train(cfg, args):   
    print(f'pytorch version: {torch.__version__}')
    print(f'Cuda is available: {torch.cuda.is_available()}')

    # for only single GPU
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]
        
    # init distributed env first, since logger depends on the dist info.        TODO : args.launcher 설정해서 사용하기 연구
    if args.launcher == 'none':
        distributed = False
    else:
        pass
        # import mvcc
        # from mmcv.runner import get_dist_info, init_dist
        # distributed = True
        # init_dist(args.launcher, **cfg.dist_params)
        # # re-set gpu_ids with distributed training mode
        # _, world_size = get_dist_info()
        # cfg.gpu_ids = range(world_size)
    
    check_data_root(cfg, args)   
    work_dir = set_dir_path(cfg)
 
    cfg.dump(os.path.join(work_dir, os.path.basename(args.cfg)))        # save config file(.py)
    
    with open(cfg.data.train.ann_file, "r") as fp:
        data_ann_file = json.load(fp)   
    json.dump(data_ann_file, open(os.path.join(work_dir, "dataset.json"), "w"), indent=4, cls=NpEncoder)        # save dataset file for test
    
    # create logger
    logger, meta, timestamp = get_logger_set_meta(work_dir, cfg, args, distributed)
    cfg.work_dir = work_dir
    
    
    # build dataset
    datasets = [build_dataset(cfg.data.train)]      # <class 'mmdet.datasets.custom.CocoDataset'>
       
    if cfg.model.type =='MaskRCNN' :
        cfg.model.roi_head.bbox_head.num_classes = len(datasets[0].CLASSES)
        cfg.model.roi_head.mask_head.num_classes = len(datasets[0].CLASSES)    
    elif cfg.model.type =='SOLOv2' :
        cfg.model.mask_head.num_classes
       
    
    
    # build model
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    if cfg.pretrained is not None:  # fine tuning
        model.init_weights()

    if cfg.checkpoint_config is not None:   
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    
    
    # train_detector
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate ,
        timestamp=timestamp,
        meta=meta)
    
    
                                                  
    
def get_logger_set_meta(work_dir, cfg, args, distributed):
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(work_dir, f'{timestamp}.log')
    logger = get_logger(name='mmdet', log_file=log_file, log_level=cfg.log_level)    

    # log env info      
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    # logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)          # delete '#' if you want print Environment info  
    # # log some basic info
    # logger.info(f'Distributed training: {distributed}')                                   # delete '#' if you want print distributed flag 
    # logger.info(f'Config:\n{cfg.pretty_text}')                                            # delete '#' if you want print config 
    
    cfg.device = get_device()                
    
    # init the meta dict to record some important information such as environment info and seed, which will be logged
    meta = dict()
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # set random seeds
    meta = set_random_seeds(cfg, args, logger, meta)
    
    return logger, meta, timestamp    


def set_random_seeds(cfg, args, logger, meta):
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = os.path.basename(args.cfg)
    return meta

    
def set_dir_path(cfg):
    yyyy_mm_dd_hhmm = time.strftime('%Y-%m-%d', time.localtime(time.time())) \
                       + "-"+ str(time.localtime(time.time()).tm_hour) \
                       + str(time.localtime(time.time()).tm_min)
                       
    os.makedirs(os.path.abspath(cfg.work_dir), exist_ok = True)
    current_work_dir =  os.path.join(os.path.abspath(cfg.work_dir), yyyy_mm_dd_hhmm + "_"+ cfg.data_category)            
    os.makedirs(current_work_dir, exist_ok = True)

    return current_work_dir
    


def check_data_root(cfg, args):
    category_dir = os.path.join(os.path.join(os.path.abspath(cfg.data_root),'train'), cfg.data_category)
    assert os.path.isdir(category_dir), f"check category train_dir path : {category_dir}"
    
    train_dataset_json = os.path.join(category_dir, cfg.train_dataset_json)
    assert os.path.isfile(train_dataset_json), f"check train dataset json file path : {train_dataset_json}"
    
  
    if args.validate:
        category_dir = os.path.join(os.path.join(os.path.abspath(cfg.data_root),'val'), cfg.data_category)
        assert os.path.isdir(category_dir), f"check category val_dir path : {category_dir}"
    
        val_dataset_json = os.path.join(category_dir, cfg.val_dataset_json)
        assert os.path.isfile(val_dataset_json), f"check val dataset json file path : {val_dataset_json}"
        
        
