import logging
from typing import List, Dict
import os
import json
import time
import glob

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
# import faiss
from sklearn import preprocessing
import cv2

###################### Yolo model ###############################
from PIL import Image
from pathlib import Path
import torch
import sys

from config.config import Config
from config.common_keys import *
from config.singleton import Singleton
from config.common_keys import *
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import check_img_size, check_suffix, non_max_suppression, scale_coords, set_logging
from yolov5.utils.torch_utils import select_device
ROOT = os.getcwd()


class CropModel(metaclass=Singleton):

    def __init__(self):
        self.config = Config()
        with open(self.config.label_file) as json_file:
            self.label_file = json.load(json_file)
        self.model_yolo_crop, self.stride, self.half, self.device = self.load_yolo_model(
            weights="yolov5/yolov5s.pt", device='cpu' if not self.config.use_gpu else 'cuda')

    @torch.no_grad()
    def load_yolo_model(self, weights=ROOT + '/yolov5s.pt', device='', half=False):
        set_logging()
        device = select_device(device)
        half &= device.type != 'cpu'
        w = str(weights[0] if isinstance(weights, list) else weights)
        classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
        check_suffix(w, suffixes)
        pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)
        stride, names = 64, [f'class{i}' for i in range(1000)]
        if pt:
            model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
            stride = int(model.stride.max())
            if half:
                model.half()
        for name in dir():
            if name not in ["model", "stride", "half", "device"]:
                del name
        return model, stride, half, device

    def choose_best_crop(self, X1, Y1, X2, Y2, confidences, crop_method=1):
        if len(confidences) == 0:
            return None, None, None, None, None
        list_index_filter_confidence = [i for i in range(len(confidences)) if confidences[i] >= self.config.crop_conf]
        if len(list_index_filter_confidence) == 0:
            list_area = [abs(X1[i] - X2[i]) * abs(Y1[i] - Y2[i]) for i in range(len(confidences))]
            max_area_index = list_area.index(max(list_area))
            return X1[max_area_index], Y1[max_area_index], X2[max_area_index], Y2[max_area_index], confidences[
                max_area_index]
        elif len(list_index_filter_confidence) == 1 and crop_method == 1:
            index = list_index_filter_confidence[0]
            return X1[index], Y1[index], X2[index], Y2[index], confidences[index]
        elif (len(list_index_filter_confidence) > 1) and crop_method == 1:
            list_area = [abs(X1[index] - X2[index]) * abs(Y1[index] - Y2[index]) for index in
                         list_index_filter_confidence]
            max_area_index = list_index_filter_confidence[list_area.index(max(list_area))]
            return X1[max_area_index], Y1[max_area_index], X2[max_area_index], Y2[max_area_index], confidences[
                max_area_index]
        elif len(list_index_filter_confidence) >= 1 and crop_method == 2:
            index = confidences.index(max(confidences))
            return X1[index], Y1[index], X2[index], Y2[index], confidences[index]

    def batch_bbox(self, batch_img: List[np.array], conf_thresh=0.2, iou_thresh=0.45, img_size=416):
        pt, max_det = True, 1000
        img_size = check_img_size(imgsz=img_size, s=self.stride)
        images = []
        for img in batch_img:
            img = LoadImages(img, img_size=img_size, stride=self.stride, auto=pt)
            images.append(img)
        im0s_shape, img_shape = None, None
        for key, i in enumerate(images):
            for _, img, im0s, _ in i:
                im0s_shape = im0s.shape
                images.pop(key)
                images.insert(key, img)
        images = np.array(images)
        img_shape = images.shape
        images = torch.from_numpy(images).to(self.device)
        images = images.half() if self.half else images.float()
        images /= 255.0
        predict = self.model_yolo_crop(images, augment=False, visualize=False)[0]
        predict = non_max_suppression(prediction=predict, conf_thres=conf_thresh,
                                      iou_thres=iou_thresh, classes=None, max_det=max_det)
        dict_bbox = {}
        for i, det in enumerate(predict):
            dict_bbox[i] = {}
            if len(det):
                det[:, :4] = scale_coords(img_shape[2:], det[:, :4], im0s_shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = int(cls)
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    if label not in dict_bbox[i]:
                        dict_bbox[i][label] = []
                    dict_bbox[i][label].append({'bounding_box': [x1, y1, x2, y2], 'confidence': conf})
        return dict_bbox

    def draw_bbox(self, image: np.ndarray, dict_bbox: Dict):
        for key, value in dict_bbox.items():
            for bbox in value:
                random_color = list(np.random.choice(range(255), size=3))
                random_color = (int(random_color[0]), int(random_color[1]), int(random_color[2]))
                bbox = bbox["bounding_box"]
                image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                      random_color, 2)
                image = cv2.putText(image, self.label_file[str(key+1)], (bbox[0], bbox[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, random_color, 2, cv2.LINE_AA)
        return image

    @staticmethod
    def draw_bbox_static(image: np.ndarray, list_bbox: List):
        random_color = list(np.random.choice(range(255), size=3))
        random_color = (int(random_color[0]), int(random_color[1]), int(random_color[2]))
        image = cv2.rectangle(image, (list_bbox[0], list_bbox[1]), (list_bbox[2], list_bbox[3]),
                              random_color, 2)
        return image

    def crop_batch(self, batch_img: List[np.array], conf_thresh=0.2, iou_thresh=0.45,):
        list_detections = self.batch_bbox(batch_img=batch_img, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
        main_box = []
        max_score = 0
        for v in list_detections.values():
            for value in v.values():
                for i in value:
                    if i["confidence"] >= max_score:
                        max_score = i["confidence"]
                        main_box = i["bounding_box"]
        return main_box


class ClothModel(metaclass=Singleton):

    def __init__(self, device=None):
        """
            Constructor for the image classifier trainer of TorchVision
        """
        self.config = Config()
        self.crop_model = CropModel()
        self.img_size = 224
        self.resize_shape = (240, 320)
        self.img_transform = self.transform(self.img_size)
        if device == None:
            self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.config.use_gpu) else "cpu")
        else:
            self.device = device
        sys.path.append('utils')
        self.model = torch.load(self.config.cloth_model_path + "/best_model.pth", map_location=self.device)
        self.model.eval()
        self.output_article = {}
        self.output_color = {}
        self.model.classifier_articleType[3].register_forward_hook(self.get_features(3, layer_name='article'))
        self.model.classifier_baseColour[3].register_forward_hook(self.get_features(3, layer_name='color'))

    @staticmethod
    def transform(img_size):
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def get_features(self, name, layer_name):
        def hook(model, input, output):
            if layer_name == 'article':
                self.output_article[name] = output.detach()
            elif layer_name == 'color':
                self.output_color[name] = output.detach()

        return hook

    def extract_image(self, img, save_preview=False):
        """
            Predict from one image at the numpy array format or string format
        """
        img, _ = self.crop_model.crop_batch([img], alpha=-0.05)
        img = img[0]
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(np.uint8(img)).convert('RGB')
        else:
            img = img
        img = img.resize(self.resize_shape)
        img = self.img_transform(img)
        if save_preview:
            pil_img = transforms.ToPILImage()(img)
            pil_img.save("preview.jpg")
        img = torch.unsqueeze(img, 0).to(self.device)
        with torch.no_grad():
            self.model(img)
            return (self.output_article[3].detach().cpu().numpy() + self.output_color[3].detach().cpu().numpy()) / 2

    def extract_image_batch(self, batch_img, save_preview=False):
        """
            Predict from one image at the numpy array format or string format
        """
        batch_img, check_clothes = self.crop_model.crop_batch(batch_img, alpha=-0.05)
        for index, img in enumerate(batch_img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize(self.resize_shape)
            img = self.img_transform(img)
            batch_img[index] = img
            if save_preview:
                pil_img = transforms.ToPILImage()(img)
                pil_img.save(f"output_video/preview_{index}.jpg")
        batch_img = torch.stack(batch_img, dim=0).to(self.device)
        with torch.no_grad():
            self.model(batch_img)
            features = (self.output_article[3].detach().cpu().numpy() + self.output_color[3].detach().cpu().numpy()) / 2
            for name in dir():
                if name != "features":
                    del name
            return features

    def compare(self, img1, img2):
        '''
            Return True if img1 and img2 are similar, otherwise False
        '''
        vector = self.extract_image_batch([img1, img2])
        cos_sim = cosine_similarity(vector[0, :].reshape(1, -1), vector[1, :].reshape(1, -1))
        if cos_sim > self.config.threshold_cloth_model:
            return cos_sim, True
        else:
            return cos_sim, False

    def gen_db(self, db):
        embedding = {}
        for dir in tqdm(os.listdir(dataset_path)):
            embedding[dir] = []
            for img_file in tqdm(glob.glob(os.path.join(dataset_path, dir, '*.jpg'))):
                vector = self.extract_image(img_file)
                vector = [list(vector[0].astype(dtype=np.float64))]
                embedding[dir].append(vector)
        with open(os.path.join(self.config.db_path, 'db.json'), 'w', encoding='utf-8') as f:
            json.dump(embedding, f)

    def load_db(self):
        with open(os.path.join(self.db_path, 'db.json'), 'r') as f:
            db = json.load(f)
        first_time = True
        list_feature = []
        list_id = []
        list_len = []
        for k, v in db.items():
            list_id.append(k)
            list_len.append(len(v))
            if first_time:
                d = np.array(v[0]).shape[1]
                print(np.array(v[0]).shape)
                index = faiss.IndexFlatIP(d)
                #
                if self.use_gpu:
                    device = faiss.StandardGpuResources()  # use a single GPU
                    index = faiss.index_cpu_to_gpu(device, 0, index)
                first_time = False
            for feature in v:
                list_feature.append(np.array(feature).astype('float32').reshape(1, 1024))
        list_feature = np.concatenate(list_feature, axis=0)
        list_feature_new = preprocessing.normalize(list_feature, norm='l2')
        index.add(list_feature_new)
        return list_len, list_id, index

    def identification(self, img, list_len, list_id, index):
        PID = None
        img = self.extract_image(img)
        max = 0
        xq = np.array(img).astype('float32').reshape(1, -1)
        xq = preprocessing.normalize(xq, norm='l2')
        start_search = time.time()
        distances, indices = index.search(xq, 1)
        print('End search: ', time.time() - start_search)
        position = indices[0][0]
        sum = 0
        for idx in range(len(list_id)):
            sum += list_len[idx]
            if position < sum:
                PID = list_id[idx]
                break
        print(f'{PID}: {distances[0][0]}')
        if distances[0][0] < self.config.threshold_cloth_model:
            return PID, distances[0][0], False
        max = distances[0][0]
        return PID, distances[0][0], True


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    df = pd.DataFrame(columns=["image", "bbox"])
    images = []
    bboxes = []
    crop_model = CropModel()
    # count = 0
    for root, dirs, files in tqdm(os.walk("/AIHCM/ComputerVision/tienhn/fashion-dataset/image/", topdown=False)):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                img = cv2.imread(file_path)
                shape = img.shape
                bbox = crop_model.crop_batch(batch_img=[img], conf_thresh=0.2, iou_thresh=0.45)
                if bbox:
                    area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
                    if 0.2*shape[0]*shape[1] < area < 0.8*shape[0]*shape[1]:
                        images.append(file_path)
                        bboxes.append(bbox)
                        # img = crop_model.draw_bbox_static(image=img, list_bbox=bbox)
                        # cv2.imwrite(f"/AIHCM/ComputerVision/tienhn/fashion-dataset/image/A/{count}.jpg", img)
                        # count += 1
            except Exception as e:
                print(e)
                print(file_path)
    df["image"] = images
    df["bbox"] = bboxes
    df.to_csv("csv_file/annotation.csv")

