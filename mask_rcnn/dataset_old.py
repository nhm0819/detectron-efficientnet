from torch.utils.data import Dataset
import cv2
import os
import numpy as np
import detectron2.data.transforms as T


class InferenceDataset(Dataset):
    def __init__(self, args, df):
        super().__init__()
        self.args = args
        self.df = df.reset_index(drop=True).copy()[::-1]
        self.num_classes = args.mask_classes
        self.transform = T.Resize((800, 1333))

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx: int):
        img_path = self.df["path"][idx]
        img_path = img_path.replace("/", "\\")
        # img = cv2.imread(img_path)
        # if img is None:
        img_array = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform.get_transform(img).apply_image(img)

        return img, img_path
