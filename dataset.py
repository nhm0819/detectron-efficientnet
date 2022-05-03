from torch.utils.data import Dataset
import os

class InferDataset(Dataset):
    def __init__(self, img_list):
        super().__init__()
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx: int):
        img_path = self.img_list[idx]

        label = 1
        if "단락" in img_path.split("\\")[-2][:2]:
            label = 0
        elif "electric" in img_path.split("\\")[-2][:2]:
            label = 0

        is_train = "test2"
        if "train" in img_path.split("\\")[-3]:
            is_train = "train"
        elif "test" in img_path.split("\\")[-3]:
            is_train = "test"

        return img_path, is_train, label



class InferDataset2(Dataset):
    def __init__(self, df, is_train="train"):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.is_train = is_train


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx: int):
        img_path = self.df["path"][idx]
        label = self.df["label"][idx]
        is_train = self.is_train

        return img_path, label, is_train
