# from detectron2.utils.logger import setup_logger
#
# setup_logger()
from detectron2 import model_zoo
from detectron2.config import get_cfg

import os, cv2
import numpy as np
from mask_rcnn.dataset import InferenceDataset, make_batch
from mask_rcnn.predictor import CustomPredictor
from torch.utils.data import DataLoader
import pandas as pd
import tqdm


def get_predictor(args):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))

    # cfg.DATASETS.TRAIN = ()
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.mask_batch
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.mask_classes

    cfg.MODEL.WEIGHTS = args.mask_best_model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    # predictor = DefaultPredictor(cfg)
    predictor = CustomPredictor(cfg, mask_id=args.mask_id)
    return predictor


def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, img_arr = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                img_arr.tofile(f)
                return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


def inference(args, predictor, data_loader, renewal=False, is_train="train"):
    print("extracting mask images for efficientnet training...")

    result = pd.DataFrame(columns=["path", "is_train", "label", "wire_score", "pred_label", "electric_output", "flame_output", "indistinct_output"])

    with tqdm.tqdm(data_loader, unit="batch") as tepoch:
        for idx, line in enumerate(tepoch):

            image_paths = line["image_path"]
            images = line["image"]
            labels = line["label"]
            # is_trains = line["is_train"]

            masks, wire_probs = predictor(images)

            for i, image_path in enumerate(image_paths):
                # file_name = os.path.basename(image_path)
                # mask_image_path = os.path.join(args.mask_folder, file_name) # args.mask_folder = os.path.join(args.file_storage_path, "masked_image")
                mask_image_path = image_path.replace("public/image", "private\\training\\original")
                mask_image_path = mask_image_path.replace("original", "original\\masked_image").replace("/", "\\")
                os.makedirs(os.path.dirname(mask_image_path), exist_ok=True)
                result = result.append({"path": image_path,
                                        "is_train": is_train,
                                        "label": labels[i],
                                        "wire_score": wire_probs[i]}, ignore_index=True)
                
                if renewal or (not os.path.isfile(mask_image_path)):
                    imwrite(mask_image_path, masks[i])
                else:
                    continue

    result_path = os.path.join(args.mask_output_dir, f"{is_train}_result.csv")
    result.to_csv(result_path, encoding="utf-8", index=False)
    
    del result


def mask_inference(args, renewal=False):
    predictor = get_predictor(args)

    # train data
    train_df = pd.read_csv(args.train_csv)
    train_dataset = InferenceDataset(args, train_df)
    train_loader = DataLoader(train_dataset, batch_size=args.mask_batch, num_workers=args.num_workers, collate_fn=make_batch, pin_memory=True)

    inference(args, predictor, train_loader, renewal=renewal, is_train="train")

    del train_df, train_dataset, train_loader

    # test data
    test_df = pd.read_csv(args.test_csv)
    test_dataset = InferenceDataset(args, test_df)
    test_loader = DataLoader(test_dataset, batch_size=args.mask_batch, num_workers=args.num_workers, collate_fn=make_batch, pin_memory=True)

    inference(args, predictor, test_loader, renewal=renewal, is_train="test")

    del test_df, test_dataset, test_loader


if __name__ == "__main__":
    mask_inference(renewal=True)