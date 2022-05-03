from sklearn.model_selection import train_test_split
import glob
import pandas as pd
import os
import csv

f = open("test_data.csv")
rdr = csv.reader(f)

lines = []
for i, line in enumerate(rdr):
    if i == 0:
        lines.append(line)
        continue
    # splits = line[0].split('\\')
    # number = int(splits[-1].split('_')[-1][:-4])
    # if splits[0] == "segmented_bad_data":
    #     if splits[1] == "flame":
    #         number = number + 10669
    #     else:
    #         number = number + 6528
    #
    # last_name = f'{number}.jpg'
    # line[0].split('_')[-1] = last_name
    #
    # line[0] = '/'.join(line[0].split('\\')[1:])
    line[0] = "test_data/" + line[0]
    lines.append(line)

write = open("write_test.csv", "w", newline="")
wr = csv.writer(write)
wr.writerows(lines)

######################################################################


df = pd.read_csv("E:\\work\\kesco\\raw_data\\20211008\\file_storage\\train_data.csv")
df["label"].value_counts()

# image paths
electric_images = glob.glob(
    "E:\\work\\kesco\\raw_data\\20211008\\segmented_bad_data\\electric\\*.jpg"
)
flame_images = glob.glob(
    "E:\\work\\kesco\\raw_data\\20211008\\segmented_bad_data\\flame\\*.jpg"
)

# aggregate
images = electric_images + flame_images
len(images)


# data frame
dict = {"electric": 0, "flame": 1}
df = pd.DataFrame(columns=["path", "label"])

for image in images:
    split_path = image.split("\\")
    image_path = os.path.join(split_path[5], split_path[6], split_path[7])
    label = image.split("\\")[6]
    df = df.append({"path": image_path, "label": dict[label]}, ignore_index=True)
len(df)

# save
df.to_csv("bad_segmented_all.csv", index=False)


# train:val:test = 8:1:1
train, val = train_test_split(
    df, test_size=0.2, random_state=42, shuffle=True, stratify=df["label"]
)
# val, test = train_test_split(val, test_size=0.5, random_state=42, shuffle=True)

train.to_csv("train_data.csv", index=False)
val.to_csv("test_data.csv", index=False)
# test.to_csv("test.csv", index=False)


import glob
import shutil

img_list1 = glob.glob(
    "E:\\work\\kesco\\raw_data\\20211008\\good_data_kor_name\\electric\\*.jpg"
)
img_list2 = glob.glob(
    "E:\\work\\kesco\\raw_data\\20211008\\bad_data_kor_name\\electric\\*.jpg"
)
img_list3 = glob.glob(
    "E:\\work\\kesco\\raw_data\\20211008\\good_data_kor_name\\flame\\*.jpg"
)
img_list4 = glob.glob(
    "E:\\work\\kesco\\raw_data\\20211008\\bad_data_kor_name\\flame\\*.jpg"
)

img_list = img_list1 + img_list2 + img_list3 + img_list4

for idx, path in enumerate(img_list):
    save_path = f"E:\\work\\kesco\\raw_data\\20211008\\file_storage\\train_data\\electric\\electric_{idx+6528}.jpg"
    shutil.copy2(path, save_path)


df["type"] = df["path"].map(lambda x: x.split("/")[0])
df["image_num"] = df["path"].apply(lambda x: x.split("_")[-1][:-4]).astype(int)
df = df.sort_values(by=["type", "image_num"])


df = df.drop(["image_path"], axis=1)

#####################
from datetime import datetime


def timestamp():
    return datetime.now().strftime("%Y-%m-%d / %Hh%Mm%Ss")


import pandas as pd

df = pd.read_csv("E:\\work\\kesco\\file_storage\\segmented_test.csv")
df["path"] = df["path"].apply(lambda x: x.replace("/", "\\"))
df["path"] = df["path"].apply(lambda x: x.replace("segmented_good_data", "test_data"))
df["path"] = df["path"].apply(lambda x: x.replace("segmented_bad_data", "test_data"))

df.to_csv("E:\\work\\kesco\\file_storage\\test.csv", index=False)


df = pd.read_csv("E:\\work\\kesco\\raw_data\\20211008\\segmented_train.csv")
df["path"] = df["path"].apply(lambda x: x.replace("/", "\\"))
df["path"] = df["path"].apply(lambda x: x.replace("segmented_good_data", "train_data"))
df["path"] = df["path"].apply(lambda x: x.replace("segmented_bad_data", "train_data"))


import pandas as pd
import glob
import os
import csv

test_electrics = glob.glob("E:\\work\\kesco\\file_storage\\test_data\\electric\\*.jpg")
test_flames = glob.glob("E:\\work\\kesco\\file_storage\\test_data\\flame\\*.jpg")
test_list = test_electrics + test_flames

df = pd.DataFrame(columns=["path", "label"])

for instance in test_list:
    path = instance[27:]
    type = instance.split("\\")[5]
    label = 0 if (type == "electric") else 1
    df = df.append({"path": path, "label": label}, ignore_index=True)

df.to_csv("E:\\work\\kesco\\file_storage\\test_data.csv", index=False)
