from torch.utils.data import Dataset
import cv2
import numpy as np
import detectron2.data.transforms as T
import torch
import os


def make_batch(samples):
    img_path = []
    label = []
    is_train = []
    image = []

    for sample in samples:
        img_path.append(sample[1])
        label.append(sample[2])
        is_train.append(sample[3])

        height = sample[0].shape[1]
        width = sample[0].shape[2]
        image_info = {"image": sample[0], "height": height, "width": width}

        image.append(image_info)

    return {
        "image": image,
        "image_path": img_path,
        "label": label,
        "is_train": is_train,
    }


class InferenceDataset(Dataset):
    def __init__(self, args, df, is_train="train"):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()[::-1]
        self.num_classes = args.mask_classes
        self.transform = T.ResizeShortestEdge(
            short_edge_length=1536, max_size=2048
        )  # T.Resize((800, 1333))
        self.is_train = is_train

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx: int):
        image_path = self.df["path"][idx].replace("/", "\\")
        label = self.df["label"][idx]
        is_train = self.is_train

        image_array = np.fromfile(image_path, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image.shape[1] > 2048:
            image = self.transform.get_transform(image).apply_image(image)

        image = torch.as_tensor(image.transpose(2, 0, 1))

        return image, image_path, label, is_train
