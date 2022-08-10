import os

from config.common_keys import *
from config.singleton import Singleton


class Config(metaclass=Singleton):
    thresh_cluster_l1 = float(os.getenv(THRESH_CLUSTER_L1, 0.97))
    use_gpu = os.getenv(USE_GPU, False)
    crop_model_path = os.getenv(CROP_MODEL_PATH, "yolov5/yolov5s.pt")
    crop_conf = os.getenv(CROP_CONF, 0.92)
    cloth_model_path = os.getenv(CLOTH_MODEL_PATH, "model/cloth_model")
    collection_path = os.getenv(COLLECTION_PATH, "mongodb://admin:admin@172.28.0.23:20253/?authSource=admin&readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false")
    mongodb_name = os.getenv(DB_NAME, "IMAGES_livestream")
    collection_name = os.getenv(COLLECTION_NAME, 'images')
    threshold_cloth_model = os.getenv(THRESHOLD_CLOTH_MODEL, 0.8)
    ip_address = os.getenv(IP_ADDRESS, "0.0.0.0")
    port = os.getenv(PORT, 35012)
    label_file = os.getenv(LABEL_FILE, "config/label.json")