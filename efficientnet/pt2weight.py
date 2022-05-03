import torch

from efficientnet.model import WireClassifier

# weigths extract
def make_eff_weights(args):
    args.clf_best_weights = args.clf_best_model.replace(".pt", ".weights")
    model = WireClassifier(num_classes=args.num_classes)

    model.load_state_dict(torch.load(args.clf_best_model))

    if 1:  # weight download, (0 -> off, 1 -> on)
        import numpy as np

        with open(args.clf_best_weights, "wb") as f:
            weights = model.state_dict()
            weight_list = [(key, value) for (key, value) in weights.items()]
            dumy = np.array([0] * 10, dtype=np.float32)
            dumy.tofile(f)

            if 0:
                for i in range(len(weight_list)):
                    key, w = weight_list[i]
                    # for k, j in enumerate(w.flatten()):
                    # print(i, key, k, j)
                exit()

            if 1:
                for idx in range(0, len(weight_list)):  # box head
                    key, w = weight_list[idx]
                    if "num_batches_tracked" in key:
                        # print(idx, "--------------------")
                        continue
                    if len(w.shape) == 2:
                        w = w.transpose(1, 0)
                        w = w.cpu().data.numpy()
                    else:
                        w = w.cpu().data.numpy()
                    w.tofile(f)
                    # print(0, idx, key, w.shape)
        print("complte to making classifier weights file.")
