from torch.utils.data import Dataset
import os
import cv2
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, args, df, transforms=None, inference=False):
        super().__init__()
        self.args = args
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.inference = inference

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx: int):
        # file_name = os.path.basename(self.df["path"][idx])
        # img_path = os.path.join(self.args.mask_folder, file_name).replace("/", "\\")
        img_path = (
            self.df["path"][idx]
            .replace("original", "original\\masked_image")
            .replace("/", "\\")
        )

        img_array = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            img = self.transforms(image=img)["image"]

        label = self.df["label"][idx]

        if self.inference:
            return img_path, img, label, idx

        return img, label
