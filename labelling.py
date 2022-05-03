import cv2
import numpy as np
import json
import os
import copy


class Labeller:

    def __init__(self, db, weight_id, nms_cnt=3, num_classes=26, wire_class_num=25,
                 image_shape=(720,1280), epsilon_rate=0.003):
        '''

        :param db: (mysql.connector.connect) DB connection
        :param weight_id: (int) group number in ln_weight_item table
        :param nms_cnt: (int) nms_count in soynet config
        :param num_classes: (int) num classes in mask rcnn
        :param wire_class_num: (int) wire class number in mask rcnn
        :param image_shape: (tuple of list) default image size is (720, 1280)
        :param epsilon_rate: (float) bigger -> make more points
                                    smaller -> make little points
        '''
        self.weight_id = weight_id
        self.db = db
        self.cursor = db.cursor()

        self.image_shape = image_shape
        self.epsilon_rate = epsilon_rate
        self.nms_cnt = nms_cnt
        self.num_classes = num_classes
        self.wire_class_num = wire_class_num

        self.json_folder = self.get_json_folder()
        self.labelling_list = self.get_labelling_list()

        with open("annotation_form.json", "r") as f:
            self.json_form = json.load(f)
            f.close()

        self.type_info = {'3': 'flame',
                          '12': 'electric',
                          '9' : 'indistinct'}

    def test(self):
        print("testing")
        return weight_id


    def get_json_folder(self):
        sql_datapath = f"select datapath from ln_weight_info where weightid={self.weight_id};"
        self.cursor.execute(sql_datapath)
        datapath_info = self.cursor.fetchall()
        data_path = datapath_info[0][0].replace("/", "\\")
        json_folder = os.path.join(data_path, "json")
        return json_folder

    def get_labelling_list(self):
        sql_labelling_list = f"select specimenid, pseq, seq, imgpath, jsonpath from ln_weight_item where weightid={self.weight_id};"
        self.cursor.execute(sql_labelling_list)
        labelling_list = self.cursor.fetchall()
        return labelling_list


    def get_bbox(self, RIP_path):
        '''
        :param RIP_path: absolute path

        :return: [x1, x2, y1, y2]
        '''
        with open(RIP_path) as f:
            bbox_list = f.readlines()
            bbox = bbox_list[0].split(",")[:4]
            bbox = np.array(bbox, dtype=float).astype(int)

        return bbox


    def get_mask_feature(self, binary_mask_path):
        '''
        :param path: binary_mask_path (absolute path)
        :param nms_cnt: nms count in soynet
        :param num_classes: number of classes in pre-trained mask rcnn
        :param wire_class: int

        :return: mask_feature -> [28, 28]
        '''
        mask_feature = np.fromfile(binary_mask_path, dtype=np.float32)
        mask_feature = mask_feature.reshape((1, self.nms_cnt, self.num_classes, 28, 28))  # mask feature shape : (28, 28)
        mask_feature = mask_feature[0, 0, self.wire_class_num, :, :]  # [first image, first index, wire_class(= 1 dim), :28, :28] --> [28, 28]

        return mask_feature


    def make_mask_image(self, mask_feature, bbox, threshold=0.5):
        '''
        :param mask_feature: [28, 28]
        :param bbox: [x1, y1, x2, y2]
        :param threshold: 0.5

        :return: mask_image -> [720, 1280]
        '''

        # get bbox
        x1, y1, x2, y2 = bbox

        # resize mask feature
        cropped_mask = cv2.resize(
            mask_feature, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        cropped_mask = np.where(cropped_mask >= threshold, 255, 0).astype(np.uint8)

        # Put the mask in the right location.
        mask_image = np.zeros(self.image_shape[:2], dtype=np.uint8)
        mask_image[y1:y2, x1:x2] = cropped_mask

        return mask_image


    def get_points(self, mask_image):
        '''
        :param mask_image: [height, width]
        :param alpha: the rate to use of arc length. (default : 0.3%)

        :return: [num_points, 1, 2], (each point has int(x) and int(y))
        '''
        ret, src = cv2.threshold(mask_image, 127, 1, 0)
        contours, hierarchy = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # RETR_CCOMP
        contour = max(contours, key=lambda x:len(x))
        epsilon = self.epsilon_rate * cv2.arcLength(contour, True)  # alpha is the rate of th arc length
        approx = cv2.approxPolyDP(contour, epsilon, True).squeeze()

        return approx.tolist()


    def autolabel(self, binary_mask_path, RIP_path):
        json_data = copy.deepcopy(self.json_form)
        try:
            bbox = self.get_bbox(RIP_path)
        except:
            print("cannot get bbox")
            return json_data

        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        size = w*h
        if size < 10000:
            print("Reject too small bounding box")
            return json_data

        try:
            mask_image = self.make_mask_image(self.get_mask_feature(binary_mask_path), bbox)
        except:
            print("cannot get mask image")
            return json_data

        points = self.get_points(mask_image)
        json_data["shapes"][0]["points"] = points

        return json_data


    def labelling(self):
        print("start labelling")

        for idx, labelling_item in enumerate(self.labelling_list):
            specimen_id = labelling_item[0]
            pseq = labelling_item[1]
            seq = labelling_item[2]
            img_path = labelling_item[3]
            check_json = labelling_item[4]

            if check_json is not None:
                continue

            # get labelling item info
            sql_labelling_info_1 = f"select maskpath, maskrip from ln_shot_history where specimenid='{specimen_id}' and pseq={pseq} and seq={seq};"
            sql_labelling_info_2 = f"select firetypecd from ln_weight_item where specimenid='{specimen_id}' and pseq={pseq} and seq={seq} and weightid={self.weight_id};"
            self.cursor.execute(sql_labelling_info_1)
            labelling_info_1 = self.cursor.fetchall()

            self.cursor.execute(sql_labelling_info_2)
            labelling_info_2 = self.cursor.fetchall()

            binary_mask_path = labelling_info_1[0][0]
            RIP_path = labelling_info_1[0][1]
            wire_type = self.type_info.get(labelling_info_2[0][0])

            if (RIP_path is None) or (binary_mask_path is None):
                continue

            if wire_type is not None: # 'flame' or 'electric' or 'indistinct'
                # labelling
                json_data = self.autolabel(binary_mask_path=binary_mask_path, RIP_path=RIP_path)

                if len(json_data["shapes"][0]["points"]) == 0:
                    print(f"idx : {idx}, "
                      f"specimenid : {labelling_item[0]}, "
                      f"is not made into labelling data")
                    continue

                json_data["imagePath"] = img_path
                json_data["shapes"][0]["type"] = wire_type

                # save labelling result to json
                json_name = os.path.basename(img_path).replace(".jpg", ".json")
                json_name = wire_type + "_" + json_name
                json_path = os.path.join(self.json_folder, json_name).replace("\\", "/")

                print(f"idx : {idx}, specimenid : {labelling_item[0]}, json_path : {json_path}")

                with open(json_path, "w", encoding='utf-8') as json_file:
                    json.dump(json_data, json_file, ensure_ascii=False)

                # update database
                sql_labelling_list = f"UPDATE ln_weight_item SET jsonpath='{json_path}' WHERE" \
                                     f" weightid = {self.weight_id} and specimenid='{specimen_id}'" \
                                     f" and pseq={pseq} and seq={seq};"

                self.cursor.execute(sql_labelling_list)
                self.db.commit()
            else:
                print(f"idx : {idx}, "
                      f"specimenid : {labelling_item[0]} is not made into labelling data")

        # db close
        self.cursor.close()



if __name__ == "__main__":
    empty_json = {
        "version": "4.5.9",
        "flags": {},
        "shapes": [
            {
                "label": "wire",
                "points": [
                ],
                "group_id": 0,
                "shape_type": "polygon",
                "flags": {}
            }
        ],
        "imagePath": "backup\\0.jpg",
        "imageHeight": 720,
        "imageWidth": 1280
    }

    ##############
    with open("infer_data\\G3FAhN3gR140bKp6HFYMxjxy2bgxmeMA_1643084025005_1_0_mask_RIP.txt") as f:
        bbox_list = f.readlines()
        bbox = bbox_list[0].split(",")[:4]
        bbox = np.array(bbox, dtype=float).astype(int)


    np_mask = np.fromfile("infer_data\\G3FAhN3gR140bKp6HFYMxjxy2bgxmeMA_1643084025005_1_0_binaryMaskData", dtype=np.float32)
    np_mask = np_mask.reshape((1,3,27,28,28))
    mask_feature = np_mask[0, 0, 26, :, :] # [num_image, first_idx, wire_class, :28, :28]

    threshold = 0.5
    x1, y1, x2, y2 = bbox

    cropped_mask = cv2.resize(
        mask_feature, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    cropped_mask = np.where(cropped_mask >= threshold, 255, 0).astype(np.uint8)

    # Put the mask in the right location.
    mask_image = np.zeros((720,1280), dtype=np.uint8)


    mask_image[y1:y2, x1:x2] = cropped_mask

    # mask_image = np.greater(mask_image, 0)
    # mask = mask_image*255 # mask.numpy()*255
    ret, src = cv2.threshold(mask_image, 127, 1, 0)
    contours, hierarchy = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #RETR_CCOMP

    epsilon = 0.003*cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)
    cv2.drawContours(mask_image, [approx], 0, (255,0,0), 2, cv2.LINE_8, hierarchy)
    cv2.imshow('dst', mask_image)

    print(approx.tolist())
    points = approx.tolist()
