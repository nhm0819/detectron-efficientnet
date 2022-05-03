import glob
import json
import shutil
import os
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np


def save_plot(output_folder, experiment_metrics):
    fig, ax1 = plt.subplots()

    color = "tab:blue"
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")

    ax1.plot(
        [x["iteration"] for x in experiment_metrics if "total_loss" in x],
        [x["total_loss"] for x in experiment_metrics if "total_loss" in x],
        color="black",
        label="Total Loss",
    )
    ax1.plot(
        [x["iteration"] for x in experiment_metrics if "validation_loss" in x],
        [x["validation_loss"] for x in experiment_metrics if "validation_loss" in x],
        color="dimgray",
        label="Val Loss",
    )

    ax1.tick_params(axis="y")
    plt.legend(loc="upper left")

    ax2 = ax1.twinx()

    color = "tab:orange"
    ax2.set_ylabel("AP")
    ax2.plot(
        # [x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
        [x["segm/AP"] for x in experiment_metrics if "segm/AP" in x],
        color=color,
        label="AP",
    )
    ax2.tick_params(axis="y")

    plt.legend(loc="upper right")
    # plt.show()

    plt.savefig(os.path.join(output_folder, "result.png"))
    plt.close()


def select_best(args):
    output_dir = args.mask_output_dir

    def load_json_arr(json_path):
        lines = []
        with open(json_path, "r") as f:
            for line in f:
                lines.append(json.loads(line))
        return lines

    experiment_metrics = load_json_arr(os.path.join(output_dir, "metrics.json"))

    # save_plot(output_dir, experiment_metrics)

    val_losses = [
        x["validation_loss"] for x in experiment_metrics if "validation_loss" in x
    ]
    segm_APs = [x["segm/AP"] for x in experiment_metrics if "segm/AP" in x]

    AP = max(segm_APs)

    # best_idx = val_losses.index(min(val_losses))
    best_idx = segm_APs.index(AP)
    best_model = os.path.join(
        output_dir, f"model_{((best_idx+1)*args.mask_checkpoint-1):07d}.pth"
    )
    save_path = os.path.join(output_dir, f"best_model_mAP_{AP:.02f}.pt")

    shutil.copy2(best_model, save_path)

    # remove_list = os.listdir(result_folder)
    remove_list = glob.glob(os.path.join(output_dir, "*.pth"))

    try:
        # remove_list.remove(save_path)
        [os.remove(no_use) for no_use in remove_list]
    except:
        print(f"Cannot remove bad models in '{output_dir}'")

    return save_path, AP


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
