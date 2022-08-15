from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests
import torch
from torchvision import transforms

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
image = transforms.Compose([transforms.ToTensor()])(image)
print(111111, image.shape, type(image))

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
inputs = feature_extractor(images=image, return_tensors="pt")
print(222222, type(inputs))

output = model(**inputs)
print(torch.equal(output.last_hidden_state[:, 0, :], output.pooler_output))


# from typing import Dict
#
# import pandas as pd
#
#
# def filter(df: pd.DataFrame, filter_dict: Dict = None, action: str = "remove"):
#     if action == "remove":
#         for column, value in filter_dict.items():
#             df = df[df[column].isin(values=value) == False]
#     elif action == "edit":
#         for column, value in filter_dict.items():
#             for i in value:
#                 df[column][df[column] == list(i.keys())[0]] = list(i.values())[0]
#     return df
#
#
# if __name__ == "__main__":
#     dict_ = {'key 1': [1, 1, 3, 3, 2, 2], 'key 2': [2, 5, 2, 2, 5, 5], 'key 3': [3, 3, 3, 3, 6, 6]}
#     dataframe = pd.DataFrame(dict_)
#     print(dataframe)
#     filter_dicts = {"key 1": [2, 3]}
#     dataframe = filter(dataframe, filter_dict=filter_dicts, action="remove")
#     print(dataframe)
#
#     dataframe = pd.DataFrame(dict_)
#     print(dataframe)
#     filter_dicts = {"key 1": [{1: 0}, {3: 100}], "key 2": [{2: 200}]}
#     dataframe = filter(dataframe, filter_dict=filter_dicts, action="edit")
#     print(dataframe)

# import numpy as np
# import random as rd
#
# from config.config import Config
#
# config_ = Config()
#
#
# def random(idx=None, is_test: bool = True):
#     if is_test:
#         if idx == 0:
#             np.random.seed(10)
#             result = np.random.choice(config_.random_neg_step) + \
#                      idx // config_.random_neg_step * config_.random_neg_step
#             return result
#         elif idx is None:
#             mix_ratio = np.random.choice(7) + 3
#             patches = mix_ratio * config_.mix_shape[0] * config_.mix_shape[1] // 10
#             patches = np.random.choice(config_.mix_shape[0] * config_.mix_shape[1], size=patches, replace=False)
#             return patches
#         else:
#             result = np.random.choice(config_.random_neg_step) + \
#                      idx // config_.random_neg_step * config_.random_neg_step
#             return result
#     else:
#         if idx is None:
#             mix_ratio = rd.choice(range(7)) + 3
#             patches = mix_ratio * config_.mix_shape[0] * config_.mix_shape[1] // 10
#             patches = rd.sample(range(config_.mix_shape[0] * config_.mix_shape[1]), patches)
#             return patches
#         else:
#             result = rd.choice(range(config_.random_neg_step)) + \
#                 idx // config_.random_neg_step * config_.random_neg_step
#             return result
#
#
# for i in range(5):
#     print(random(i, is_test=False))
#
# print("*" * 20)
#
# for i in range(5):
#     print(random(is_test=False))
#
# print("*"*20)
#
# for i in range(5):
#     print(random(i, is_test=True))
#
# print("*" * 20)
#
# for i in range(5):
#     print(random(is_test=True))