# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
from torch.utils.data import Dataset

from custom_mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .pipelines import Compose

from .api_wrappers import COCO

@DATASETS.register_module()
class CustomDataset(Dataset):
    """Custom dataset for detection.

    The annotation format is shown as follows. The `ann` field is optional for
    testing.

    .. code-block:: none

        [
            {
                'filename': 'a.jpg',
                'width': 1280,
                'height': 720,
                'ann': {
                    'bboxes': <np.ndarray> (n, 4) in (x1, y1, x2, y2) order.
                    'labels': <np.ndarray> (n, ),
                    'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                    'labels_ignore': <np.ndarray> (k, 4) (optional field)
                }
            },
            ...
        ]

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        data_root (str, optional): Data root for ``ann_file``,
            ``img_prefix``, ``seg_prefix``, ``proposal_file`` if specified.
        test_mode (bool, optional): If set True, annotation will not be loaded.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes of the dataset's classes will be filtered out. This option
            only works when `test_mode=False`, i.e., we never filter images
            during tests.
    """

    CLASSES = None    
        
    PALETTE = None

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 file_client_args=dict(backend='disk')):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.file_client = mmcv.FileClient(**file_client_args)
        self.data_ann = mmcv.load(ann_file) 
        self.CLASSES = self.get_classes(self.data_ann, classes)
        self.PALETTE = self.get_palette()
 
        # print(f"self.ann_file : {self.ann_file}")
        # print(f"self.data_root : {self.data_root}")
        # print(f"self.img_prefix : {self.img_prefix}")
        # print(f"self.proposal_file : {self.proposal_file}")
        # print(f"self.test_mode : {self.test_mode}")
        # print(f"self.filter_empty_gt : {self.filter_empty_gt}")
        # print(f"self.file_client : {self.file_client}")
        # print(f"self.CLASSES : {self.CLASSES}")
        # exit()

        
        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
                
        
        # load annotations (and proposals)
        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(self.ann_file) as local_path:
                self.data_infos = self.load_annotations(local_path)
  
        else:
            warnings.warn(
                'The used MMCV version does not have get_local_path. '
                f'We treat the {self.ann_file} as local paths and it '
                'might cause errors if the path is not a local path. '
                'Please use MMCV>= 1.3.16 if you meet errors.')
            self.data_infos = self.load_annotations(self.ann_file)

        if self.proposal_file is not None:  # 해당 없음
            if hasattr(self.file_client, 'get_local_path'):
                with self.file_client.get_local_path(
                        self.proposal_file) as local_path:
                    self.proposals = self.load_proposals(local_path)
            else:
                warnings.warn(
                    'The used MMCV version does not have get_local_path. '
                    f'We treat the {self.ann_file} as local paths and it '
                    'might cause errors if the path is not a local path. '
                    'Please use MMCV>= 1.3.16 if you meet errors.')
                self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None
                
        # filter images too small and containing no annotations
        if not test_mode:  
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()
            
        # processing pipeline
        self.pipeline = Compose(pipeline)
       
        

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)


    def get_palette(self):
        if self.CLASSES is not None:
            # just for visualization. no relationship with classification loss.
            palette = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                    (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
                    (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
                    (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
                    (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
                    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
                    (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
                    (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
                    (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
                    (134, 134, 103), (145, 148, 174), (255, 208, 186),
                    (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
                    (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
                    (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
                    (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
                    (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
                    (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
                    (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
                    (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
                    (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
                    (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
                    (191, 162, 208)]
            
            tmp_palette = []
            for i in range(len(self.CLASSES)):
                tmp_palette.append(palette[i])
            return tmp_palette
        return None
    
    
    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
      

        # for using custom dataset
        data_ann = mmcv.load(ann_file)                                          #
        if data_ann['info']['description'] == 'Hibernation Custom Dataset':     #
            self.cat_ids = []                                                   #
            for cat_dict in data_ann['categories']:                             #
                self.cat_ids.append(cat_dict['id'])                             #
        else :                                                                  #
            self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
    
        data_infos = []
        total_ann_ids = []
        for d, i in enumerate(self.img_ids):
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)                     # [{'license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id', 'filename'}]
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)

        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        
        return data_infos
    

    def load_proposals(self, proposal_file):
        """Load proposal from proposal file."""
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)


    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann


    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]
    

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds
    

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """ 
        self.flag = np.zeros(len(self), dtype=np.uint8)

        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    @classmethod
    def get_classes(cls, data_ann, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:   
            classes = data_ann['classes']
            return classes             
            

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def get_cat2imgs(self):
        """Get a dict with class as key and img_ids as values, which will be
        used in :class:`ClassAwareSampler`.

        Returns:
            dict[list]: A dict of per-label image list,
            the item of the dict indicates a label index,
            corresponds to the image index that contains the label.
        """
        if self.CLASSES is None:
            raise ValueError('self.CLASSES can not be None')
        # sort the label index
        cat2imgs = {i: [] for i in range(len(self.CLASSES))}
        for i in range(len(self)):
            cat_ids = set(self.get_cat_ids(i))
            for cat in cat_ids:
                cat2imgs[cat].append(i)
        return cat2imgs

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results

    def __repr__(self):
        """Print the number of instance number."""
        # return dataset to (build_from_cfg) of (build_dataset) in dataset>builder.py
        dataset_type = 'Test' if self.test_mode else 'Train'
        result = (f'\n{self.__class__.__name__} {dataset_type} dataset '
                  f'with number of images {len(self)}, '
                  f'and instance counts: \n')
        
        if self.CLASSES is None:
            result += 'Category names are not provided. \n'
            return result
        
        instance_count = np.zeros(len(self.CLASSES) + 1).astype(int)
    
        # count the instance number in each image
        for idx in range(len(self)):
            label = self.get_ann_info(idx)['labels']  
            unique, counts = np.unique(label, return_counts=True)  
            if len(unique) > 0:
                # add the occurrence number to each class
                instance_count[unique] += counts
            else:
                # background is the last index
                instance_count[-1] += 1   
                
        # create a table with category count
        table_data = [['category', 'count'] * 5]
        row_data = []

        for cls, count in enumerate(instance_count):
            if cls < len(self.CLASSES):
                row_data += [f'{cls} [{self.CLASSES[cls]}]', f'{count}']
            else:
                # add the background number
                row_data += ['-1 background', f'{count}']  
            if len(row_data) == 10:
                table_data.append(row_data)
                row_data = []
        if len(row_data) >= 2:
            if row_data[-1] == '0':
                row_data = row_data[:-2]
            if len(row_data) >= 2:
                table_data.append([])
                table_data.append(row_data)

    
        table = AsciiTable(table_data)
        result += table.table        
        return result
