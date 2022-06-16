

from ast import arg
import mmcv
from mmcv import Config
import argparse
import os

import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is not True!"

# python main.py --mode labelme --cfg configs/labelme_config.py --ann paprika
# python main.py --mode train --cfg configs/train_config.py --cat test

def set_config(args):
    cfg = Config.fromfile(args.cfg)
    
    assert isinstance(cfg, mmcv.Config), \
        f'cfg got wrong type: {type(cfg)}, expected mmcv.Config'
    
    if args.mode == 'labelme':
        if args.json_name is not None: 
            if os.path.splitext(args.json_name)[1] !=".json":
                raise IOError('Only .json type are supoorted now!')
            cfg.json.file_name = args.json_name
            
        if args.ann not in cfg.json.valid_categorys:
            raise KeyError(f"{args.ann} is not valid category.")
        else: cfg.json.category = args.ann
        
    elif args.mode == 'train':
        # set dataset path
        if args.root is not None:       cfg.data_root = args.root
        cfg.data_category = args.cat
        if args.train_json is not None:       cfg.dataset_json = args.train_json

        cfg.data.train.ann_file = cfg.data_root + "/" + cfg.data_category + "/" + cfg.dataset_json
        cfg.data.train.img_prefix= cfg.data_root + "/" + cfg.data_category + '/'
        cfg.data.val.ann_file, cfg.data.test.ann_file = cfg.data.train.ann_file, cfg.data.train.ann_file
        cfg.data.val.img_prefix, cfg.data.test.img_prefix = cfg.data.train.img_prefix, cfg.data.train.img_prefix
        
        # set work_dir path
        if args.work_dir is not None:   cfg.work_dir = args.work_dir

        
        
        # TODO : add val
    return cfg

def parse_args():
    parser = argparse.ArgumentParser(description="labelme annotation to custom data json file.")
    
    parser.add_argument('--mode', required = True, choices=['labelme', 'train', 'inference'])
    parser.add_argument("--cfg", required = True, help="name of config file")
    
    # mode : labelme 
    parser.add_argument('--ann', help= "category of dataset")
    parser.add_argument('--json_name', help="name of json file")
    
    # mode : train
    parser.add_argument('--root', help = 'root dir path of dataset')
    parser.add_argument('--cat', help = 'name of ceragory dir of dataset')
    parser.add_argument('--train_json', help= "name of dataset json file")
    
    parser.add_argument('--work_dir', help= "name of working dir (save env, log text and config .py file)")
    
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    
    
    
    args = parser.parse_args()
    
    
    if args.mode == 'labelme':
        assert args.ann is not None, "'--ann' is required if mode is labelme."
    elif args.mode == 'train':
        assert args.cat is not None, "'--cat' is required if mode is train."
        
    
    return args



if __name__ == "__main__":

    args = parse_args()
    
    cfg = set_config(args)
    
    if args.mode == 'labelme':
        from labelme import labelme_custom 
        labelme_custom(cfg, args)
    
    if args.mode == 'train':
        from train import train
        
        train(cfg, args)
        pass
        