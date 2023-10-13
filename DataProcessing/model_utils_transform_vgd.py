import logging
from typing import List, Dict
import os
import json
import time
import glob
import random as rd
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
import requests
from io import BytesIO
import base64
from torchvision import transforms
from sklearn.utils import shuffle

cwd = os.getcwd().replace("\\", "/")
cwd = "/".join(cwd.split("/")[:-1])
sys.path.append(cwd)
from config.config_vit import Config
from services.transform import RandomDraw, MultiQuality
config_ = Config()


url = "http://172.29.13.24:35515/process/vg_api"
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2


def get_base64(file_name):
    img = Image.open(file_name) # path to file
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data) # bytes
    base64_str = base64_str.decode("utf-8") # str
    return base64_str


def show_demo(file_path, text, output_path=None, file_name=None):
    image = Image.open(file_path)
    base64_str = get_base64(file_path)
    payload = "{\r\n    \"index\": \"a;lkalsd\",\r\n    \"data\": [{\r\n        \"image\": \"" + base64_str + "\",\r\n        \"text\": \"" + text + "\"\r\n    }]\r\n} "
    headers = {'Content-Type': 'application/json'}

    response = requests.request("POST", url, headers=headers, data=payload)
    response = response.json()
    start_point = response['data'][0]['start_point']
    end_point = response['data'][0]['end_point']
    # img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    # cv2.rectangle(
    #     img,
    #     (start_point[0], start_point[1]),
    #     (end_point[0], end_point[1]),
    #     (0, 255, 0),
    #     3
    # )
    # img = cv2.putText(img, text, org, font, 
    #                 fontScale, color, thickness, cv2.LINE_AA)
    # cv2.imwrite(f'{output_path}/{file_name}', img)
    return [start_point[0], start_point[1], end_point[0], end_point[1]]


def draw_bbox_static(image: np.ndarray, list_bbox: List):
    random_color = list(np.random.choice(range(255), size=3))
    random_color = (int(random_color[0]), int(random_color[1]), int(random_color[2]))
    image = cv2.rectangle(image, (list_bbox[0], list_bbox[1]), (list_bbox[2], list_bbox[3]),
                            random_color, 3)
    return image


def augmentation(image: Image) -> Image:
    w, h = image.size
    ratio_random_crop = np.random.choice(range(70, 90)) / 100
    h = int(h * ratio_random_crop)
    w = int(w * ratio_random_crop)
    pad_value = int(np.random.choice(range(20, 30)) / 100 * w)
    padding_transform = [
        transforms.Pad(pad_value, fill=0, padding_mode="constant"),
        transforms.Pad(pad_value // 2, padding_mode="edge"),
        # transforms.Pad(pad_value // 2, padding_mode="reflect"),
    ]
    padding_transform = padding_transform[np.random.choice(len(padding_transform))]
    list_transform = [
        padding_transform,
        transforms.RandomCrop(size=(h, w)),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    ]
    transform = transforms.Compose([
        MultiQuality(random_ratio=0.2),
        RandomDraw(random_ratio=0.5),
        transforms.RandomCrop(size=(h, w)),
        transforms.RandomAutocontrast(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(transforms=np.random.choice(list_transform, 2), p=0.8)
    ])
    image = transform(image)
    return image


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    # df1 = pd.read_csv("csv_file/annotation_shopee_v3.csv", index_col=False)
    # df2 = pd.read_csv("csv_file/annotation_shopee_v3_augmentation.csv", index_col=False)
    # df3 = pd.read_csv("csv_file/annotation_shopee_v3_augmentation_1.csv", index_col=False)
    # valid = {"original": 0, "one": 0, "two": 0}
    # for i in range(len(df1)):
    #     if "augmentation" in df2.iloc[i]["image"] and "augmentation" in df3.iloc[i]["image"]:
    #         valid["two"] += 1
    #     elif "augmentation" in df2.iloc[i]["image"] or "augmentation" in df3.iloc[i]["image"]:
    #         valid["one"] += 1
    #     else:
    #         valid["original"] += 1

    df = pd.read_csv("csv_file/annotation_shopee_v3.csv", index_col=False)
    img_augment_1 = df["image"].tolist()
    img_augment_2 = df["image"].tolist()
    bbox_1 = df["bbox"].tolist()
    bbox_2 = df["bbox"].tolist()
    for i in tqdm(range(len(df) // 2)):
        try:
            df.iloc[i, 1] = df.iloc[i, 1].replace("/AIHCM/ComputerVision/tienhn/fashion-dataset", "/AIHCM/FileShare/Public/AI_Member/tienhn/server23/fashion-dataset")
            image = Image.open(df.iloc[i]["image"])
            if np.random.choice(2) == 1:
                file_path = f"../image_augmentation_v2/{i}_1.jpg"
                image_1 = augmentation(image=image)
                image_1 = image_1.convert('RGB')
                image_1.save(file_path)
                image_1 = cv2.imread(file_path)
                shape = image_1.shape
                bbox = show_demo(
                    file_path=file_path, text="the most important thing", 
                    output_path=f"../image_augmentation_v2/image_vgd", file_name=f"{i}_1.jpg")
                if bbox:
                    area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) 
                    if 0.1*shape[0]*shape[1] < area < 0.99*shape[0]*shape[1]:
                        bbox_1[i] = str(bbox)
                        img_augment_1[i] = file_path
            if np.random.choice(2) == 1:
                file_path = f"../image_augmentation_v2/{i}_2.jpg"
                image_2 = augmentation(image=image)
                image_2 = image_2.convert('RGB')
                image_2.save(file_path)
                image_2 = cv2.imread(file_path)
                shape = image_2.shape
                bbox = show_demo(
                    file_path=file_path, text="the most important thing", 
                    output_path=f"../image_augmentation_v2/image_vgd", file_name=f"{i}_2.jpg")
                if bbox:
                    area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
                    if 0.1*shape[0]*shape[1] < area < 0.99*shape[0]*shape[1]:
                        bbox_2[i] = str(bbox)
                        img_augment_2[i] = file_path
        except Exception as e:
            print(e)

    df["image_1"] = img_augment_1
    df["bbox_1"] = bbox_1
    df["image_2"] = img_augment_2
    df["bbox_2"] = bbox_2
    df.to_csv("csv_file/annotation_shopee_v3_augmentation_v2.csv", index=False)

    # df = pd.read_csv("/AIHCM/ComputerVision/tienhn/fashion-dataset/TokenMix/DataProcessing/csv_file/annotation_shopee_v3.csv")
    # for i in tqdm(range(len(df))):
    #     list_bbox = df.iloc[i]["bbox"][1:-1].split(",")
    #     list_bbox = [int(i.strip()) for i in list_bbox]
    #     image = cv2.imread(df.iloc[i]["image"])
    #     image = draw_bbox_static(image=image, list_bbox=list_bbox)
    #     cv2.imwrite(f"/AIHCM/ComputerVision/tienhn/fashion-dataset/TokenMix/DataProcessing/image_test/{i}.jpg", image)
    
    # df = pd.DataFrame(columns=["image", "bbox"])
    # images = []
    # bboxes = []
    # count = 0
    # for root, dirs, files in tqdm(os.walk("/AIHCM/ComputerVision/tienhn/fashion-dataset/image/", topdown=False)):
    #     for name in files:
    #         count += 1
    #         file_path = os.path.join(root, name)
    #         try:
    #             pos_img = Image.open(file_path)
    #             img = augmentation(image=img)
    #             shape = img.shape
    #             bbox = show_demo(
    #                 file_path=file_path, text="the most important thing", 
    #                 output_path=f"/AIHCM/ComputerVision/tienhn/fashion-dataset/image/A", file_name=f"{count}.jpg")
    #             if bbox:
    #                 area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
    #                 if 0.1*shape[0]*shape[1] < area < 0.9*shape[0]*shape[1]:
    #                     images.append(file_path)
    #                     bboxes.append(bbox)
    #         except Exception as e:
    #             print(e)
    #             print(file_path)
    # df["image"] = images
    # df["bbox"] = bboxes
    # df.to_csv("csv_file/annotation_shopee_v3.csv")
