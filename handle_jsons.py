import json
import glob
import os
import shutil


if __name__ == "__main__":
    json_list = glob.glob("Z:\\private\\training\\original\\test_data\\json\\*.json")

    for idx, json_path in enumerate(json_list):
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)

            json_data["imagePath"] = os.path.join("Z:\\private\\training\\original\\test_data", json_data["imagePath"])

            save_path = json_path

            with open(save_path, 'w') as f:
                json.dump(json_data, f, indent=4)
