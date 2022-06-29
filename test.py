import os
import time
import glob
import cv2
from tqdm import tqdm 
import random

import mmcv
# from mmcv.runner import init_dist, get_dist_info, _load_checkpoint
from mmcv.runner import init_dist, get_dist_info, load_checkpoint

import torch
import warnings

from custom_mmdet.apis import multi_gpu_test, single_gpu_test
from custom_mmdet.models import build_detector
from custom_mmdet.utils import compat_cfg, setup_multi_processes, get_device, build_ddp, build_dp
from custom_mmdet.datasets import replace_ImageToTensor, build_dataloader, build_dataset 
                                               
from custom_mmdet.apis import init_detector, inference_detector
from custom_mmdet.core import encode_mask_results


def test(cfg, args):
    
    print(f'pytorch version: {torch.__version__}')
    print(f'Cuda is available: {torch.cuda.is_available()}')
    
    cfg = compat_cfg(cfg)
    path_dict = check_set_dir_root(cfg, args)
   
    # set multi-process settings
    # setup_multi_processes(cfg)
    
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)


    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)
    
    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    
    
    

    
    batch_size = 4
    imgs_path = glob.glob(os.path.join(path_dict['test_images_dir'], "*.jpg"))
    batch_imgs = [imgs_path[x:x + batch_size] for x in range(0, len(imgs_path), batch_size)]
    
    cfg.model.train_cfg = None
    # model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))     # just build model, why needed?
    
    device = 'cuda:0'
    model = init_detector(cfg, path_dict['model_file'], device = device)

    outputs = []
    if not distributed:
        codel_config = model.cfg  
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        model.cfg = codel_config

        for imgs_path in tqdm(batch_imgs):   
            with torch.no_grad():
                results = inference_detector(model, imgs_path)

            out_files = []
            for img_path in imgs_path:
                file_name = os.path.basename(img_path)
                out_file = os.path.join(path_dict['result_img_dir'], file_name)
                out_files.append(out_file)
            
            for img_path, out_file, result in zip(imgs_path, out_files, results):
                img = cv2.imread(img_path)      
    
                model.module.show_result(
                        img, 
                        result,
                        bbox_color= (0, 255, 255),
                        text_color= (255, 255, 255),
                        mask_color= 'random',
                        show=True,
                        out_file=out_file,
                        score_thr=args.show_score_thr)
                      
            # encode mask results
            if isinstance(result[0], tuple):
                results = [(bbox_results, encode_mask_results(mask_results))
                        for bbox_results, mask_results in results]
            # This logic is only used in panoptic segmentation test.
            elif isinstance(results[0], dict) and 'ins_results' in results[0]:
                for j in range(len(results)):
                    bbox_results, mask_results = results[j]['ins_results']
                    results[j]['ins_results'] = (bbox_results,
                                                encode_mask_results(mask_results))     
        
            outputs.extend(results)
    else:   # TODO
        # model = build_ddp(
        #     model,
        #     cfg.device,
        #     device_ids=[int(os.environ['LOCAL_RANK'])],
        #     broadcast_buffers=False)
        # outputs = multi_gpu_test(model, data_loader, args.tmpdir,
        #                          args.gpu_collect)
        pass
        
    
    # # -------------
    # test_loader_cfg = {
    #     **test_dataloader_default_args,
    #     **cfg.data.get('test_dataloader', {})
    # }
        
    # # build the dataloader
    # dataset = build_dataset(cfg.data.test)
    # data_loader = build_dataloader(dataset, **test_loader_cfg)
    
    
    # # build the model and load checkpoint
    # cfg.model.train_cfg = None
    # model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))     # just build model, why needed?
    
    # # from mmcv.runner import load_checkpoint 
    # # checkpoint = load_checkpoint(model, model_path, map_location='cpu')
    
    # checkpoint = load_checkpoint(model, path_dict['model_file'], map_location='cpu')
    # # checkpoint = _load_checkpoint(path_dict['model_file'], map_location='cpu')
   
    
    # if 'CLASSES' in checkpoint.get('meta', {}):
    #     model.CLASSES = checkpoint['meta']['CLASSES']
    # else:
    #     model.CLASSES = dataset.CLASSES

    # if not distributed:
    #     model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
   
    #     outputs = single_gpu_test(model, data_loader, 
    #                               out_dir = path_dict['result_img_dir'],
    #                               img_dir = path_dict['test_images_dir'],
    #                               show_score_thr = args.show_score_thr)
    # else:   # TODO
    #     model = build_ddp(
    #         model,
    #         cfg.device,
    #         device_ids=[int(os.environ['LOCAL_RANK'])],
    #         broadcast_buffers=False)
    #     outputs = multi_gpu_test(model, data_loader, args.tmpdir,
    #                              args.gpu_collect)


    rank, _ = get_dist_info()   
    
    if rank == 0:
        print(f"\n writing results to {path_dict['pkl_file']}")
        mmcv.dump(outputs, path_dict['pkl_file'])   
            
            
        kwargs = {} 

        # TODO 
        # if args.eval:       # python main.py --mode test --cfg configs/train_config.py --model_dir 2022-06-22-1457_paprika --cat paprika --epo 40 --eval mAP
        #     eval_kwargs = cfg.get('evaluation', {}).copy()
        #     # hard-code way to remove EvalHook args
        #     for key in [
        #             'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
        #             'rule', 'dynamic_intervals'
        #     ]:
        #         eval_kwargs.pop(key, None)
        #     eval_kwargs.update(dict(metric=args.eval, **kwargs))
        #     metric = dataset.evaluate(outputs, **eval_kwargs)
        #     print(metric)
        #     metric_dict = dict(config=args.config, metric=metric)
        #     if path_dict['result_dir'] is not None and rank == 0:
        #         mmcv.dump(metric_dict, path_dict['eval_file'])
      

def check_set_dir_root(cfg, args):
    path_dict = {}
    
    # check test images path
    test_images_dir_path = os.path.join(cfg.data.test.img_prefix)
    assert os.path.isdir(test_images_dir_path), f"check test images directory path : {test_images_dir_path}"
        
    # check model path
    model_dir = os.path.join(cfg.work_dir, args.model_dir)
    assert os.path.isdir(model_dir), f"check model directory path : {model_dir}"
        
    # check model path
    if args.epo == "latest": 
        model_path = os.path.join(model_dir, args.epo + ".pth")
    else:
        model_path = os.path.join(model_dir, "epoch_" + args.epo + ".pth")
    assert os.path.isfile(model_path), f"check model file path : {model_path}"
    
    
    # check test dataset path
    category_dir = os.path.join(os.path.join(os.path.abspath(cfg.data_root),'test'), cfg.data_category)
    assert os.path.isdir(category_dir), f"check category dir path : {category_dir}"
    
    # set directoris
    yyyy_mm_dd_hhmm = time.strftime('%Y-%m-%d', time.localtime(time.time())) \
                       + "-"+ str(time.localtime(time.time()).tm_hour) \
                       + str(time.localtime(time.time()).tm_min)
    yyyy_mm_dd_hhmm_cat =  yyyy_mm_dd_hhmm + "_" + args.cat             
    
    result_dir = os.path.join(model_dir, yyyy_mm_dd_hhmm_cat)
    os.makedirs(result_dir, exist_ok=True)
    result_images_dir = os.path.join(result_dir, cfg.show_dir)
    os.makedirs(result_images_dir, exist_ok=True)
   
    pkl_path = os.path.join(result_dir, cfg.output)
    
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    eval_file = os.path.join(result_dir, f'eval_{timestamp}.json')
    
    path_dict['test_images_dir'] = test_images_dir_path
    path_dict['model_file'] = model_path
    path_dict['result_dir'] = result_dir
    path_dict['result_img_dir'] = result_images_dir
    path_dict['pkl_file'] = pkl_path
    path_dict['eval_file'] = eval_file
    
      
    return path_dict
    
