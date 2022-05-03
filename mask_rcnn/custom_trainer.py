from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    DatasetMapper,
)
import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils

from mask_rcnn.loss_eval_hook import LossEvalHook

import torch
import os
import copy
import numpy as np


class MyColorAugmentation(T.Augmentation):
    def get_transform(self, image):
        r = np.random.rand(2)
        return T.ColorTransform(lambda x: x * r[0] + r[1] * 10)


def custom_mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    # T.Resize((800,800)),
    transform_list = [
        # T.Resize((720, 1280)),
        T.ResizeShortestEdge(
            short_edge_length=(640, 672, 704, 736, 768, 800),
            max_size=1333,  # (640, 672, 704, 736, 768, 800)
            sample_style="choice",
        ),
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        MyColorAugmentation(),
        T.RandomBrightness(0.8, 1.2),
        T.RandomSaturation(0.8, 1.2),
        T.RandomContrast(0.8, 1.2),
        T.RandomRotation([-90, 90]),
    ]

    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    # @classmethod
    # def build_train_loader(cls, cfg):
    #     return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=[
    #         # T.Resize((1280,720)),
    #         T.ResizeShortestEdge(short_edge_length=(640, 672, 704, 720), max_size=1280,
    #         sample_style='choice'),
    #         T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
    #         T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
    #         T.RandomBrightness(0.8, 1.2),
    #         T.RandomSaturation(0.8, 1.2),
    #         T.RandomContrast(0.7, 1.2)
    # ]))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(
            -1,
            LossEvalHook(
                self.cfg.TEST.EVAL_PERIOD,
                self.model,
                build_detection_test_loader(
                    self.cfg, self.cfg.DATASETS.TEST[0], DatasetMapper(self.cfg, True)
                ),
            ),
        )
        return hooks
