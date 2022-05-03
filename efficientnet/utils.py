import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_transforms(img_size, data):
    if data == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
                A.Flip(),
                A.Transpose(),
                A.OneOf(
                    [A.GaussNoise(), A.NoOp(), A.MultiplicativeNoise(), A.ISONoise()],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.MotionBlur(p=0.2),
                        A.MedianBlur(blur_limit=3, p=0.1),
                        A.Blur(blur_limit=3, p=0.1),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.OpticalDistortion(p=0.3),
                        A.GridDistortion(p=0.1),
                        A.PiecewiseAffine(p=0.3),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.CLAHE(clip_limit=2),
                        A.Sharpen(),
                        A.Emboss(),
                        A.RandomBrightnessContrast(),
                        A.ColorJitter(
                            brightness=0.2, contrast=0.1, saturation=0, hue=0
                        ),
                        A.RandomGamma(),
                    ],
                    p=0.5,
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.Rotate(limit=45, p=0.5),
                ToTensorV2(always_apply=True),
            ]
        )

    elif data == "test":
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(always_apply=True),
            ]
        )


import os
import glob
import shutil
from efficientnet.pt2weight import make_eff_weights


def eff_select_best(args):
    # model clean up
    model_list = glob.glob(os.path.join(args.eff_output_dir, "*.pt"))
    acc_list = [float(name.split("_")[-1][:-3]) for name in model_list]
    best_acc = max(acc_list)
    best_idx = acc_list.index(best_acc)
    best_model = model_list[best_idx]
    model_name = f"best_model_acc_{best_acc}.pt"
    save_path = os.path.join(args.eff_output_dir, model_name)
    args.clf_best_model = save_path
    shutil.copy2(best_model, save_path)

    # remove other weights
    [os.remove(no_use) for no_use in model_list]

    # pt to weights
    make_eff_weights(args)
