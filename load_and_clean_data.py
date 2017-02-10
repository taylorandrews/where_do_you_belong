import numpy as np
import pandas as pd
import glob
from string import maketrans
import itertools
import string

def get_features():
    init_features = []
    for idx, filename in enumerate(glob.glob('data/*.csv')):
        init_features.append(filename[5:-4])
    return init_features

def build_column(feature):
    filename = str('data/' + feature + '.csv')
    feat_data = pd.read_csv(filename)
    feat_data.drop('HDI Rank', axis=1, inplace=True)
    feat_data_np = np.array(feat_data)
    feat_data_cols = np.array(feat_data.columns[1:], dtype=str)
    feat_data_rows = np.core.defchararray.translate(np.char.lstrip(np.char.lower(np.array(feat_data.pop('Country'), dtype='|S'))), maketrans(' ', '_'), "()'.-/'\x994567890")
    rows = np.array((['_'.join(row) for row in np.array(list(itertools.product(feat_data_rows, feat_data_cols)))]))
    a=string.maketrans('','')
    nodigs=a.translate(a, '0123456789.')
    feat_data_np = np.array(feat_data)
    flattened=[]
    np.array([[flattened.append(float(elem.translate(a, nodigs))) if type(elem) == str else flattened.append(elem) for elem in row] for row in feat_data_np])
    # print type(flattened[1])
    # print np.array(flattened, dtype=float)
    return np.vstack((rows, np.array(flattened, dtype=float))).T

def all_idxs(dfs):
    master = []
    [[master.append(idx) for idx in df[0] if idx not in master] for df in dfs]
    return sorted(master)

def make_dfs(init_features):
    dfs = []
    for feature in init_features:
        data = build_column(feature)
        dfs.append(pd.DataFrame(data))
    return dfs

def final_dataframe(dfs, master, init_features):
    data = np.array(master)
    for df in dfs:
        col = np.array([])
        i = 0
        for idx in master:
            if idx not in np.array(df[0]):
                col = np.append(col, np.nan)
            else:
                col = np.append(col, np.array(df[1][i]))
                i += 1
        data = np.vstack((data, col))
    final_rows = data[0]
    pure_data = np.delete(data, 0, 0).T
    return pd.DataFrame(data=pure_data, index=final_rows, columns=init_features, dtype=float)

if __name__ == '__main__':

    init_features = get_features()
    dfs = make_dfs(init_features)
    master = all_idxs(dfs)
    X = final_dataframe(dfs, master, init_features)
    y = X.pop('HDI')
