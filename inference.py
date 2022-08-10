from urllib.request import urlopen

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from config.config import Config
from main import Model

config_ = Config()


class Inference:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def read_img(self, input_image, transform=None):
        image = None
        if isinstance(input_image, str):
            try:
                image = Image.open(urlopen(input_image))
            except Exception as e:
                image = Image.open(input_image)
        elif isinstance(input_image, np.ndarray):
            image = Image.fromarray(input_image)
        elif isinstance(input_image, Image.Image):
            image = input_image
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(self.config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        image = transform(image)
        image = torch.unsqueeze(image, 0)
        return image

    def inference(self, image, model_path: str = "Checkpoints/weight/resnet_model_0_0.0714285746216774.pth"):
        image = self.read_img(input_image=image)
        model = torch.load(model_path).to(self.device)
        model.eval()
        with torch.no_grad():
            image = image.to(self.device)
            return model.forward_one(image)


if __name__ == "__main__":
    import warnings
    import os
    warnings.filterwarnings("ignore")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config_.device
    results = Inference(config=config_).inference(image="pos.jpg")
    print(results.shape)
