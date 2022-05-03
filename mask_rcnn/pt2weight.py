import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo


def make_mask_weights(args):
    OPTS = ["MODEL.WEIGHTS", args.mask_best_model]
    args.mask_best_weight = args.mask_best_model.replace(".pt", ".weights")
    cfg = get_cfg()  # get a fresh new config
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
        )
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.mask_classes  # 학습 시킨 클래스 개수
    cfg.merge_from_list(OPTS)
    predictor = DefaultPredictor(cfg)

    with open(args.mask_best_weight, "wb") as f:
        weight_list = [
            (key, value) for (key, value) in predictor.model.state_dict().items()
        ]
        dumy = np.array([0] * 10, dtype=np.float32)
        dumy.tofile(f)
        if 0:
            for i in range(len(weight_list)):
                key, w = weight_list[i]
                # print(0, i, key, w.shape)

                # if abs(j + 0.0098662) < 0.0000001 :
                #     print(i, key, k,j)
            exit()
        if len(weight_list) == 562:  # .pt
            for idx in range(16, 536):  # resnet
                key, w = weight_list[idx]
                w = w.cpu().numpy()
                w.tofile(f)
                # print(0, idx, key, w.shape)

            for idx in range(12, -1, -4):  # fpn
                for idx_2 in range(4):
                    key, w = weight_list[idx + idx_2]
                    w = w.cpu().numpy()
                    w.tofile(f)
                    # print(0, idx + idx_2, key, w.shape)

            for r_idx in range(5):  # rpn * 5 반복
                for idx in range(536, 542):
                    key, w = weight_list[idx]
                    w = w.cpu().numpy()
                    w.tofile(f)
                    # print(0, idx, key, w.shape)

            for idx in range(542, 562):  # box head
                key, w = weight_list[idx]
                w = w.cpu().numpy()

                if len(w.shape) == 2:
                    w = np.transpose(w, (1, 0))
                w.tofile(f)
                # print(0, idx, key, w.shape)

        elif len(weight_list) < 567:  # 버리는 값이 없어 짧음
            for idx in range(16, 536):  # resnet
                key, w = weight_list[idx]
                w.cpu().numpy().tofile(f)
                # print(0, idx, key, w.shape)

            for idx in range(12, -1, -4):  # fpn
                for idx_2 in range(4):
                    key, w = weight_list[idx + idx_2]
                    w.cpu().numpy().tofile(f)
                    # print(0, idx + idx_2, key, w.shape)

            for r_idx in range(5):  # rpn * 5 반복
                for idx in range(541, 547):
                    key, w = weight_list[idx]
                    w.cpu().numpy().tofile(f)
                    # print(0, idx, key, w.shape)

            for idx in range(547, 555):  # box head
                key, w = weight_list[idx]
                w = w.cpu().numpy()

                if len(w.shape) == 2:
                    w = np.transpose(w, (1, 0))
                w.tofile(f)
                # print(0, idx, key, w.shape)

        else:
            for idx in range(16, 536):  # resnet
                key, w = weight_list[idx]
                w.cpu().numpy().tofile(f)
                # print(0, idx, key, w.shape)

            for idx in range(12, -1, -4):  # fpn
                for idx_2 in range(4):
                    key, w = weight_list[idx + idx_2]
                    w.cpu().numpy().tofile(f)
                    # print(0, idx+idx_2, key, w.shape)

            for r_idx in range(5):  # rpn * 5 반복
                for idx in range(541, 547):
                    key, w = weight_list[idx]
                    w.cpu().numpy().tofile(f)
                    # print(0, idx, key, w.shape)

            for idx in range(547, 555):  # box head
                key, w = weight_list[idx]
                w = w.cpu().numpy()
                if len(w.shape) == 2:
                    w = np.transpose(w, (1, 0))
                w.tofile(f)
                # print(0, idx, key, w.shape)

            for idx in range(555, 567):  # mask_head
                key, w = weight_list[idx]
                w.cpu().numpy().tofile(f)
                # print(0, idx, key, w.shape)

        print("complete to making mask weights file.")
