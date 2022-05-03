import argparse
import os
import pandas as pd
import cv2
import torch
import albumentations as A
import shutil
import torch.nn.functional as F

from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
from efficientnet.dataset import CustomDataset
from efficientnet.model import WireClassifier


def inference(args, clf, is_train="train"):

    # transform
    transform = A.Compose(
        [
            A.Resize(height=380, width=380),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    # dataset
    result_path = os.path.join(args.mask_output_dir, f"{is_train}_result.csv")
    df = pd.read_csv(result_path).copy()
    dataset = CustomDataset(args, df, transform, inference=True)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # data load
    with torch.no_grad():
        with tqdm(loader, unit="batch") as tepoch:
            for batch_idx, (image_path, image, label, idx) in tqdm(enumerate(tepoch)):

                # inference
                input = image.to(args.device)
                pred = clf(input)

                pred_scores = F.softmax(pred)
                pred_labels = pred_scores.max(1)[1].to("cpu")
                pred_scores = pred_scores.to("cpu").numpy()

                for i, df_idx in enumerate(idx.numpy()):
                    pred_label = pred_labels[i].item()
                    output_0 = pred_scores[i][0].item()
                    output_1 = pred_scores[i][1].item()
                    output_2 = pred_scores[i][2].item()

                    df.loc[
                        df_idx,
                        [
                            "pred_label",
                            "electric_output",
                            "flame_output",
                            "indistinct_output",
                        ],
                    ] = (pred_label, output_0, output_1, output_2)

    try:
        os.remove(result_path)
    except:
        pass
    result_path = os.path.join(args.eff_output_dir, f"{is_train}_result.csv")
    df.to_csv(result_path, encoding="utf-8", index=False)


def eff_inference(args):

    # model init
    try:
        clf = WireClassifier(num_classes=args.num_classes)
        clf.load_state_dict(torch.load(args.clf_best_model))
    except:
        clf = torch.jit.load(args.clf_best_model, map_location=args.device)
    clf.to(args.device)
    clf.eval()

    inference(args, clf, is_train="train")
    inference(args, clf, is_train="test")
