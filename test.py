import numpy as np
import random as rd

from config.config import Config

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


for i in range(5):
    print(random(i, is_test=False))

print("*" * 20)

for i in range(5):
    print(random(is_test=False))

print("*"*20)

for i in range(5):
    print(random(i, is_test=True))

print("*" * 20)

for i in range(5):
    print(random(is_test=True))

