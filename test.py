import os
import time

import mmcv
from mmcv.runner import init_dist, get_dist_info, load_checkpoint

import torch
import warnings

from custom_mmdet.apis import multi_gpu_test, single_gpu_test
from custom_mmdet.models import build_detector
from custom_mmdet.utils import compat_cfg, setup_multi_processes, get_device, build_ddp, build_dp
from custom_mmdet.datasets import replace_ImageToTensor, build_dataloader, build_dataset 
                                               


def test(cfg, args):
    
    print(f'pytorch version: {torch.__version__}')
    print(f'Cuda is available: {torch.cuda.is_available()}')
    
    cfg = compat_cfg(cfg)
    model_path, output_file_path, eval_file = check_set_dir_root(cfg, args)
    
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

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
        
    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))     # TODO 이거 뜯어고쳐서 model을 registry에서 꺼내 올 필요성이 있는지 확인할 것 
    
    checkpoint = load_checkpoint(model, model_path, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)



    
        
    rank, _ = get_dist_info()   

    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, eval_file)
        


def check_set_dir_root(cfg, args):

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
   
    output_file_path = os.path.join(result_dir, cfg.output)
    
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    eval_file = os.path.join(result_dir, f'eval_{timestamp}.json')
    
    
    return model_path, output_file_path, eval_file
    
