from config.singleton import Singleton


class Config(metaclass=Singleton):
    # Data
    save_img = False
    image_size = 256
    # Object is divided 7*7 patch
    mix_shape = (7, 7)
    # Range of dataset to choose the negative image
    random_neg_step = 500
    shuffle = True
    test_data_size = 0.1

    # Model
    epochs = 200
    batch_size = 12
    similarity_thresh = 0.1
    device = "3"
    tensorboard_dir = "Checkpoints/tensorboard2"
    save_model_dir = "Checkpoints/weight2"
    save_model_period = 1
    lr = 0.001

