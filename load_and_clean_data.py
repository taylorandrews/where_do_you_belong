import numpy as np
import pandas as pd
import glob

def get_features():
    init_features = []
    for idx, filename in enumerate(glob.glob('data/*.csv')):
        init_features.append(filename[5:-4])
    return init_features

def build_column(feature):
    filename = str('data/' + feature + '.csv')
    feat_data = pd.read_csv(filename)
    print feat_data


if __name__ == '__main__':

    init_features = get_features()

    build_column('population')

    # for feature in init_features:
    #     build_column(feature)
