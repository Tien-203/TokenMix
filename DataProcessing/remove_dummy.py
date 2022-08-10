import glob
import os

import cv2

def remove_dummy(folder: str):
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                cv2.imread(file_path).shape
            except Exception as e:
                print(e)
                print(file_path)
                os.remove(file_path)

if __name__ == "__main__":
    remove_dummy("/AIHCM/ComputerVision/tienhn/fashion-dataset/image")