"""将经纬度对应至网格点"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import os


class GridSearch(object):
    def __init__(self):
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), "hexagon_grid_table.csv"),names=['grid_id', 'lat_0',
                         'lng_0', 'lat_1', 'lng_1', 'lat_2', 'lng_2', 'lat_3', 'lng_3', 'lat_4', 'lng_4', 'lat_5', 'lng_5'])
        df["grid_id"] = list(range(len(df)))

        self.vert = df[['lat_0', 'lng_0', 'lat_1', 'lng_1', 'lat_2', 'lng_2', 'lat_3', 'lng_3', 'lat_4', 'lng_4', 'lat_5', 'lng_5']].to_numpy().reshape((len(df),6,2))
        self.centorids = self.vert.mean(axis=1)
        self.grid_idx = list(df['grid_id'].values)
        self.tree = cKDTree(self.centorids)

    def cal_loc_grid(self, lat_lng):
        lat_lng = np.array(lat_lng)
        _,idx = self.tree.query(lat_lng, 1)
        return list(map(self.grid_idx.__getitem__, list(idx)))
