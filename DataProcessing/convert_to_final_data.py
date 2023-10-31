import regex

import numpy as np
import pandas as pd

np.random.seed(10)


def convert_to_final_data(input_file: str):
    df = pd.read_csv(input_file, index_col=False)
    pattern = r"\d_\d.jpg$"
    for i in range(len(df)):
        if regex.search(pattern, df.iloc[i, 3]) and regex.search(pattern, df.iloc[i, 5]):
            pass
        elif regex.search(pattern, df.iloc[i, 3]):
            if np.random.choice(2) == 1:
                df.iloc[i, 3] = df.iloc[i, 1]
                df.iloc[i, 4] = df.iloc[i, 2]
                print(i, df.iloc[i, 3])
    df.to_csv(f"/AIHCM/ComputerVision/tienhn/fashion-dataset/TokenMix/DataProcessing/csv_file/data_test_and_train.csv", index=False)


if __name__=="__main__":
    convert_to_final_data(input_file="/AIHCM/ComputerVision/tienhn/fashion-dataset/TokenMix/DataProcessing/csv_file/annotation_shopee_v3_augmentation.csv")
