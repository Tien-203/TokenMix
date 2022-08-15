from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import pandas as pd
import numpy as np
import random as rd
import cv2
from PIL import Image
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from transformers import ViTModel, ViTFeatureExtractor

from config.config_vit import Config
from config.common_keys import *

config_ = Config()


def random(idx=None, is_test: bool = True):
    if is_test:
        if idx == 0:
            np.random.seed(10)
            result = np.random.choice(config_.random_neg_step) + \
                     idx // config_.random_neg_step * config_.random_neg_step
            return result
        elif idx is None:
            mix_ratio = np.random.choice(7) + 3
            patches = mix_ratio * config_.mix_shape[0] * config_.mix_shape[1] // 10
            patches = np.random.choice(config_.mix_shape[0] * config_.mix_shape[1], size=patches, replace=False)
            return patches
        else:
            result = np.random.choice(config_.random_neg_step) + \
                     idx // config_.random_neg_step * config_.random_neg_step
            return result
    else:
        if idx is None:
            mix_ratio = rd.choice(range(7)) + 3
            patches = mix_ratio * config_.mix_shape[0] * config_.mix_shape[1] // 10
            patches = rd.sample(range(config_.mix_shape[0] * config_.mix_shape[1]), patches)
            return patches
        else:
            result = rd.choice(range(config_.random_neg_step)) + \
                idx // config_.random_neg_step * config_.random_neg_step
            return result


class FashionDataLoader(Dataset):
    def __init__(self, df: pd.DataFrame, config: Config, transform=None, is_test: bool = True):
        self.config = config
        self.df = df
        self.transform = transform
        self.is_test = is_test
        self.idx = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pos_idx = idx
        neg_idx = random(idx, is_test=self.is_test)
        if neg_idx >= len(self):
            neg_idx -= len(self)
        pos_sample = self.get_single_image(pos_idx)
        neg_sample = self.get_single_image(neg_idx)
        mix_sample, pos_label, neg_label = self.token_mix(pos_sample=pos_sample.copy(), neg_sample=neg_sample.copy())
        if self.config.save_img:
            img = self.draw_bbox_static(
                image=np.array(pos_sample[IMAGE]), list_bbox=pos_sample[BBOX])
            cv2.imwrite("pos.jpg", img)
            img = self.draw_bbox_static(
                image=np.array(neg_sample[IMAGE]), list_bbox=neg_sample[BBOX])
            cv2.imwrite("neg.jpg", img)
            img = self.draw_bbox_static(
                image=np.array(mix_sample[IMAGE]), list_bbox=mix_sample[BBOX])
            cv2.imwrite("mix.jpg", img)
        if self.transform:
            pos_sample[IMAGE] = self.transform(pos_sample[IMAGE])
            neg_sample[IMAGE] = self.transform(neg_sample[IMAGE])
            mix_sample[IMAGE] = self.transform(mix_sample[IMAGE])
        return {
            POS_SAMPLE: pos_sample,
            NEG_SAMPLE: neg_sample,
            MIX_SAMPLE: mix_sample,
            POS_RATIO: pos_label,
            NEG_RATIO: neg_label
        }

    def get_single_image(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.df.iloc[idx, 0]
        bbox = self.df.iloc[idx, 1][1:-1]
        bbox = [int(i) for i in bbox.split(", ")]
        image = cv2.imread(img_path)
        shape = image.shape
        height_scale = self.config.image_size / shape[0]
        width_scale = self.config.image_size / shape[1]
        bbox = list(np.array(bbox) * np.array([width_scale, height_scale, width_scale, height_scale]))
        bbox = [int(i) for i in bbox]
        image = Image.fromarray(cv2.resize(image, (self.config.image_size, self.config.image_size)))
        sample = {IMAGE: image, BBOX: bbox}
        return sample

    def token_mix_box_region(self, pos_sample: Dict, neg_sample: Dict):
        # Ex: mix_ratio=2 => take random 4 patches from neg_img replace to pos_img (5*4 patches each image)
        mix_ratio = np.random.choice(11)
        patches = mix_ratio * self.config.mix_shape[0] * self.config.mix_shape[1] // 10
        patches = np.random.choice(self.config.mix_shape[0] * self.config.mix_shape[1], size=patches, replace=False)
        mix_img = np.array(pos_sample[IMAGE])
        pos_img = mix_img[pos_sample[BBOX][1]:pos_sample[BBOX][3], pos_sample[BBOX][0]:pos_sample[BBOX][2]]
        neg_img = np.array(neg_sample[IMAGE])
        neg_img = neg_img[neg_sample[BBOX][1]:neg_sample[BBOX][3], neg_sample[BBOX][0]:neg_sample[BBOX][2]]
        neg_img = cv2.resize(neg_img, (pos_img.shape[1], pos_img.shape[0]))

        mask_img = np.zeros(pos_img.shape)
        x_step = pos_img.shape[0] // self.config.mix_shape[0]
        y_step = pos_img.shape[1] // self.config.mix_shape[1]
        for patch in patches:
            patch = np.unravel_index(patch, self.config.mix_shape)
            y0 = y_step * patch[1]
            x0 = x_step * patch[0]
            y1 = min(y_step * (patch[1] + 1), pos_img.shape[1])
            x1 = min(x_step * (patch[0] + 1), pos_img.shape[0])
            mask_img[x0:x1, y0:y1, :] = 255
        if self.config.save_img:
            cv2.imwrite("a.jpg", mask_img)
        mask_img = np.where(mask_img == 255, neg_img, pos_img)
        mix_img[pos_sample[BBOX][1]:pos_sample[BBOX][3], pos_sample[BBOX][0]:pos_sample[BBOX][2]] = mask_img
        mix_img = Image.fromarray(mix_img)
        mix_sample = {IMAGE: mix_img, BBOX: pos_sample[BBOX]}
        return mix_sample

    def token_mix(self, pos_sample: Dict, neg_sample: Dict):
        # Ex: mix_ratio=2 => take random 4 patches from neg_img replace to pos_img (5*4 patches each image)
        patches = random(is_test=self.is_test)

        pos_sample[IMAGE] = np.array(pos_sample[IMAGE])
        neg_sample[IMAGE] = np.array(neg_sample[IMAGE])
        neg_sample[IMAGE] = cv2.resize(neg_sample[IMAGE], (pos_sample[IMAGE].shape[1], pos_sample[IMAGE].shape[0]))
        height_scale = pos_sample[IMAGE].shape[0] / neg_sample[IMAGE].shape[0]
        width_scale = pos_sample[IMAGE].shape[1] / neg_sample[IMAGE].shape[1]
        bbox = list(np.array(neg_sample[BBOX]) * np.array([width_scale, height_scale, width_scale, height_scale]))
        bbox = [int(i) for i in bbox]
        neg_sample[BBOX] = bbox

        mask_img = np.zeros(pos_sample[IMAGE].shape)
        x_step = pos_sample[IMAGE].shape[0] // self.config.mix_shape[0]
        y_step = pos_sample[IMAGE].shape[1] // self.config.mix_shape[1]
        for patch in patches:
            patch = np.unravel_index(patch, self.config.mix_shape)
            y0 = y_step * patch[1]
            x0 = x_step * patch[0]
            y1 = min(y_step * (patch[1] + 1), pos_sample[IMAGE].shape[1])
            x1 = min(x_step * (patch[0] + 1), pos_sample[IMAGE].shape[0])
            mask_img[x0:x1, y0:y1, :] = 255
        if self.config.save_img:
            cv2.imwrite("a.jpg", mask_img)
        roi_mask = mask_img[pos_sample[BBOX][1]:pos_sample[BBOX][3], pos_sample[BBOX][0]:pos_sample[BBOX][2]][:, :, 0]
        pos_label = round((roi_mask == 0).sum() / roi_mask.shape[0] / roi_mask.shape[1], 1)
        roi_mask = mask_img[neg_sample[BBOX][1]:neg_sample[BBOX][3], neg_sample[BBOX][0]:neg_sample[BBOX][2]][:, :, 0]
        neg_label = round((roi_mask == 255).sum() / roi_mask.shape[0] / roi_mask.shape[1], 1)
        mix_img = np.where(mask_img == 255, neg_sample[IMAGE], pos_sample[IMAGE])
        mix_img = Image.fromarray(mix_img)
        mix_sample = {IMAGE: mix_img, BBOX: pos_sample[BBOX]}
        return mix_sample, torch.tensor(float(pos_label)), \
               torch.tensor(float(neg_label))

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx < len(self):
            result = self.get_single_image(self.idx)
            self.idx += 1
            return result
        else:
            raise "Index out of range dataset"

    @staticmethod
    def draw_bbox_static(image: np.ndarray, list_bbox: List):
        random_color = list(np.random.choice(range(255), size=3))
        random_color = (int(random_color[0]), int(random_color[1]), int(random_color[2]))
        image = cv2.rectangle(image, (list_bbox[0], list_bbox[1]), (list_bbox[2], list_bbox[3]),
                              random_color, 2)
        return image


class Model(nn.Module):
    def __init__(self, config: Config = None):
        super(Model, self).__init__()
        self.vit_16 = ViTModel.from_pretrained(config.pretrained)
        self.cosine = nn.CosineSimilarity()

    def forward_one(self, x):
        x = self.vit_16(**x).pooler_output
        return x

    def forward(self, pos_img, neg_img, mix_img):
        pos_feature = self.forward_one(pos_img)
        neg_feature = self.forward_one(neg_img)
        mix_feature = self.forward_one(mix_img)
        pos_diff = self.cosine(pos_feature, mix_feature)
        neg_diff = self.cosine(neg_feature, mix_feature)
        return pos_diff, neg_diff


class TrainModel:
    def __init__(self, csv_file: str, config: Config):
        self.csv_file = csv_file
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Model(config=config).to(self.device)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(config.pretrained)
        self.criterion = nn.MSELoss().to(self.device)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, betas=(0.9, 0.999),
        #                             eps=1e-08, weight_decay=0, amsgrad=False)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5, eta_min=0.00001)
        self.train_data, self.test_data = self.train_test_split(csv_file=csv_file)
        print(self.train_data.iloc[0])
        self.train_data = self.data_loader(self.train_data, is_test=False,
                                           transform=transforms.Compose([
                                               transforms.Resize(self.config.image_size),
                                               transforms.ToTensor(),
                                               transforms.RandomAutocontrast(),
                                               transforms.RandomHorizontalFlip(p=0.5),
                                               transforms.RandomCrop(size=int(self.config.image_size*0.9))]))
        self.test_data = self.data_loader(self.test_data, is_test=True)
        self.writer = SummaryWriter(log_dir=self.config.tensorboard_dir, flush_secs=2)

    def data_loader(self, df: pd.DataFrame, is_test, transform=None):
        if transform is None:
            data = FashionDataLoader(df=df, config=self.config, is_test=is_test,
                                     transform=transforms.Compose([
                                         transforms.Resize(self.config.image_size),
                                         transforms.ToTensor()]))
        else:
            data = FashionDataLoader(df=df, config=self.config, is_test=is_test, transform=transform)
        return DataLoader(data, batch_size=self.config.batch_size, num_workers=1)

    def train_test_split(self, csv_file):
        df = pd.read_csv(csv_file)
        df.dropna(axis=1, how="any", inplace=True)
        df = df.drop(columns=[df.columns[0]])
        if self.config.shuffle:
            df = shuffle(df, random_state=10)
        train_df, test_df = train_test_split(df, test_size=self.config.test_data_size, random_state=10)
        return train_df, test_df

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        train_accuracy = 0
        test_loss = None
        test_accuracy = None
        for iter, data in tqdm(enumerate(self.train_data)):
            if iter % 500 and iter > 0:
                a = list(self.model.parameters())[-1].clone()
            self.optimizer.zero_grad()
            pos_img = self.feature_extractor(images=[img for img in data[POS_SAMPLE][IMAGE]],
                                             return_tensors="pt").to(self.device)
            neg_img = self.feature_extractor(images=[img for img in data[NEG_SAMPLE][IMAGE]],
                                             return_tensors="pt").to(self.device)
            mix_img = self.feature_extractor(images=[img for img in data[MIX_SAMPLE][IMAGE]],
                                             return_tensors="pt").to(self.device)
            pos_diff, neg_diff = self.model(pos_img, neg_img, mix_img)
            pos_label = data[POS_RATIO].to(self.device)
            neg_label = data[NEG_RATIO].to(self.device)
            loss = (self.criterion(pos_diff, pos_label) + self.criterion(neg_diff, neg_label)) / 2
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            train_loss += loss
            accuracy = (torch.abs(pos_diff - pos_label) <= self.config.similarity_thresh).sum() + \
                       (torch.abs(neg_diff - neg_label) <= self.config.similarity_thresh).sum()
            train_accuracy += accuracy
            if iter % 500 == 0 and iter > 0:
                acc = train_accuracy/((iter+1)*self.config.batch_size*2)
                print(f"\t\tEpoch {epoch} - iter {iter} - train_loss {loss} - train_acc {acc}")
                b = list(self.model.parameters())[-1].clone()
                print(torch.equal(b.data, a.data))
        train_loss /= len(self.train_data)
        train_accuracy = train_accuracy / len(self.train_data) / self.config.batch_size / 2
        print(f"Train_loss: {train_loss} - Train_acc: {train_accuracy}")
        self.writer.add_scalar("Train Loss", train_loss, epoch)
        self.writer.add_scalar("Train Accuracy", train_accuracy, epoch)
        if epoch % self.config.save_model_period == 0:
            print("Staring validation...")
            test_loss = 0
            test_accuracy = 0
            self.model.eval()
            with torch.no_grad():
                for iter, data in tqdm(enumerate(self.test_data)):
                    pos_img = self.feature_extractor(images=[img for img in data[POS_SAMPLE][IMAGE]],
                                                     return_tensors="pt").to(self.device)
                    neg_img = self.feature_extractor(images=[img for img in data[NEG_SAMPLE][IMAGE]],
                                                     return_tensors="pt").to(self.device)
                    mix_img = self.feature_extractor(images=[img for img in data[MIX_SAMPLE][IMAGE]],
                                                     return_tensors="pt").to(self.device)
                    pos_diff, neg_diff = self.model(pos_img, neg_img, mix_img)
                    pos_label = data[POS_RATIO].to(self.device)
                    neg_label = data[NEG_RATIO].to(self.device)
                    loss = (self.criterion(pos_diff, pos_label) + self.criterion(neg_diff, neg_label)) / 2
                    test_loss += loss
                    accuracy = (torch.abs(pos_diff - pos_label) <= self.config.similarity_thresh).sum() + \
                               (torch.abs(neg_diff - neg_label) <= self.config.similarity_thresh).sum()
                    test_accuracy += accuracy
            test_loss /= len(self.test_data)
            test_accuracy = test_accuracy / len(self.test_data) / self.config.batch_size / 2
            self.writer.add_scalar("Test Loss", test_loss, epoch)
            self.writer.add_scalar("Test Accuracy", test_accuracy, epoch)
            print(f"Test_loss: {test_loss} - Test_acc: {test_accuracy}")
        return {
            TRAIN_LOSS: train_loss,
            TRAIN_ACCURACY: train_accuracy,
            TEST_LOSS: test_loss,
            TEST_ACCURACY: test_accuracy
        }

    def train(self):
        max_acc = 0
        for epoch in range(self.config.epochs):
            print(f"Starting epoch {epoch}...")
            result = self.train_epoch(epoch=epoch)
            if result[TEST_ACCURACY] is not None and result[TEST_ACCURACY] > max_acc:
                if not os.path.exists(self.config.save_model_dir):
                    os.mkdir(self.config.save_model_dir)
                torch.save(self.model, f"{self.config.save_model_dir}/resnet_model_{epoch}_{result[TEST_ACCURACY]}.pth")
                print(f"Save model at {self.config.save_model_dir}/resnet_model_{epoch}_{result[TEST_ACCURACY]}.pth")
                max_acc = result[TEST_ACCURACY]


if __name__ == "__main__":
    import warnings
    import os
    warnings.filterwarnings("ignore")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config_.device
    TrainModel(csv_file="DataProcessing/csv_file/annotation.csv", config=config_).train()

