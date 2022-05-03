from detectron2.utils.logger import setup_logger

setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


import os, cv2
import numpy as np
from mask_rcnn.dataset import InferenceDataset
from torch.utils.data import DataLoader
import pandas as pd
import tqdm


def get_predictor(args):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
        )
    )

    # cfg.DATASETS.TRAIN = ()
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.mask_classes

    cfg.MODEL.WEIGHTS = args.mask_best_model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    predictor = DefaultPredictor(cfg)
    return predictor


def load_mask(args, predictor, image):
    # im = cv2.imread(image)
    outputs = predictor(image)

    output = outputs["instances"].to("cpu")
    wire_idxs = (output.pred_classes == args.mask_id).nonzero().squeeze()

    if wire_idxs.nelement() == 0:
        return image
    elif wire_idxs.nelement() == 1:
        wire_idx = wire_idxs.item()
    else:
        wire_idx = wire_idxs.numpy()[0]

    mask = output.pred_masks[wire_idx]

    mask = np.greater(mask, 0)  # get only non-zero positive pixels/labels
    mask = np.expand_dims(mask, axis=-1)  # (H, W) -> (H, W, 1)
    mask = np.concatenate((mask, mask, mask), axis=-1)
    bit_and = np.multiply(image, mask)
    mask_0 = bit_and
    where_0 = np.where(mask_0 == 0)
    mask_0[where_0] = 255

    try:
        x1, y1, x2, y2 = (
            output.pred_boxes.tensor[wire_idx].squeeze().numpy().astype(int)
        )
        return mask_0[y1:y2, x1:x2]
    except:
        return mask_0


def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, img_arr = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode="w+b") as f:
                img_arr.tofile(f)
                return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


def inference(args, predictor, data_loader, type="train", renewal=False):
    print("extracting mask images for efficientnet training...")
    with tqdm.tqdm(data_loader, unit="batch") as tepoch:
        for idx, (image, image_path) in enumerate(tepoch):

            # file_name = "\\".join(image_path[0].split("\\")[-3:])
            file_name = os.path.basename(image_path[0])
            mask_image_path = os.path.join(args.mask_folder, file_name)

            if renewal:
                mask = load_mask(args, predictor, image.data.numpy()[0])
            else:
                if os.path.isfile(mask_image_path):
                    continue
                else:
                    mask = load_mask(args, predictor, image.data.numpy()[0])

            # cv2.imwrite(mask_image_path, mask[..., ::-1])
            imwrite(mask_image_path, mask)


def mask_inference(args, renewal=False):
    predictor = get_predictor(args)

    # train data
    train_df = pd.read_csv(args.train_csv)
    train_dataset = InferenceDataset(args, train_df)
    data_loader = DataLoader(train_dataset, batch_size=1, num_workers=args.num_workers)

    inference(args, predictor, data_loader, type="train", renewal=renewal)

    # test data
    test_df = pd.read_csv(args.test_csv)
    test_dataset = InferenceDataset(args, test_df)
    data_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers)

    inference(args, predictor, data_loader, type="test", renewal=renewal)
