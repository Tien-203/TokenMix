from PIL import Image, ImageDraw
import cv2
import numpy as np


class RandomDraw:
    def __init__(self, random_ratio: float = 0.2):
        self.color_list = ["red", "black", "blue", "green", "yellow", "purple", "gray"]
        self.width = 5
        self.random_ratio = random_ratio
        np.random.seed(10)

    def __call__(self, sample: Image) -> Image:
        is_process = np.random.choice(2, p=[1 - self.random_ratio, self.random_ratio])
        if is_process:
            w, h = sample.size
            list_points = self.linespace(h, w)
            color = np.random.choice(self.color_list)
            draw = ImageDraw.Draw(sample)
            for i in range(len(list_points) - 1):
                draw.line([list_points[i], list_points[i + 1]], width=self.width, fill=color)  
        return sample

    def linespace(self, h, w):
        num_points = np.random.choice(40) + 20
        x = np.random.choice(w//2)
        y = np.random.choice(h)
        list_points = [(x, y)]
        for i in range(num_points):
            x, y = list_points[-1]
            x += np.random.choice(20)
            y += np.random.choice(60) - 30
            list_points.append((x, y))
        return list_points


if __name__ == "__main__":
    img = Image.open("/AIHCM/ComputerVision/tienhn/fashion-dataset/TokenMix/9.jpg")
    transform = RandomDraw()
    for i in range(20):
        a = transform(img)
    a.save("a.jpg")