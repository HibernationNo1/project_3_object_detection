

from ast import arg
import mmcv
from mmcv import Config
import argparse
import os

import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is not True!"

# python main.py --mode labelme --cfg configs/labelme_config.py --ann paprika
# python main.py --mode train --cfg configs/train_config.py --cat paprika --epo 40
# python main.py --mode test --cfg configs/test_config.py --model_dir 2022-06-22-1457_paprika --cat paprika --epo 40

def set_config(args):
    cfg = Config.fromfile(args.cfg)
    
    assert isinstance(cfg, mmcv.Config), \
        f'cfg got wrong type: {type(cfg)}, expected mmcv.Config'
        
    assert cfg.mode == args.mode, f"commend mode is '{args.mode}', but you call '{cfg.mode}' config."
    
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
        cfg.data.train.ann_file = cfg.data_root + "/train/" + cfg.data_category + "/" + cfg.dataset_json
        cfg.data.train.img_prefix= cfg.data_root + "/train/" + cfg.data_category + '/'
        
        cfg.data.val.ann_file, cfg.data.val.img_prefix = cfg.data.train.ann_file, cfg.data.train.img_prefix
        
        # set work_dir path
        if args.work_dir is not None:   cfg.work_dir = args.work_dir
        if args.epo is not None: cfg.runner.max_epochs = args.epo        
        
        # TODO : add val
    
    elif args.mode == 'test':
        if args.root is not None:       cfg.data_root = args.root
        cfg.data_category = args.cat
        cfg.data.test.img_prefix = cfg.data_root + "/test/" + cfg.data_category + '/' 
        cfg.data.test.ann_file  = os.path.join(os.path.join(os.path.abspath(cfg.work_dir), args.model_dir), cfg.dataset_json ) 
       
        if args.work_dir is not None:   cfg.work_dir = args.work_dir
    
    return cfg

def parse_args():
    parser = argparse.ArgumentParser(description="Change structure from comments to custom dataset in json file format.")
    
    parser.add_argument('--mode', required = True, choices=['labelme', 'train', 'test'])
    parser.add_argument("--cfg", required = True, help="name of config file")
    
    # mode : labelme 
    parser.add_argument('--ann', help= "category of dataset     \n required")
    parser.add_argument('--json_name', help="name of train dataset file in .json format")
    parser.add_argument('--train', action = 'store_true', help = 'if True, go training after make custom dataset' )
    
    
    # mode : train
    parser.add_argument('--train_json', help= "name of train dataset file in .json format")
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
        
    # mode : test
    parser.add_argument('--model_dir', help='directory name containing trained model in .pth format  \n required')
    
    
    # mode train or test
    parser.add_argument('--work_dir', help= "name of working dir (save env, log text and config .py file)")
    parser.add_argument('--root', help = 'root dir path of dataset')
    parser.add_argument('--cat', help = 'mode-train: name of ceragory dir of train dataset, \n mode-test: name of ceragory dir of test dataset  \n required')
    parser.add_argument('--epo', type= int, help= "mode-train: max epoch, \n mode-test : model name in .pth format,  usually epoch number")
 
       
    # 
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
    
    
    args = parser.parse_args()
    
    
    if args.mode == 'labelme':
        assert args.ann is not None, "'--ann' is required if mode is labelme."
    elif args.mode == 'train':
        assert args.epo is not None, "'--epo' is required if mode is train."
        # assert args.cat is not None, "'--cat' is required if mode is train."
        
    elif args.mode == 'test':
        assert args.model_dir is not None, "'--model_dir' is required if mode is test."
        if args.epo is None : args.epo = 'latest'
        else : args.epo = str(args.epo)
    
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
    
    if args.mode == 'test':
        from test import test
        
        test(cfg, args)
        pass
        