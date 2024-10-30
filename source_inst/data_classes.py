#  CREATE CLASS FOR LOADING AND PREPROCESSING DATASET

# import libraries
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.transforms import Compose, NormalizeScale, RandomFlip, NormalizeFeatures, KNNGraph, RadiusGraph
import pandas as pd
import warnings
import zipfile
import numpy as np
from copy import copy
from re import search
from os import remove, chdir
from os.path import split, splitext
from functools import reduce
from collections import OrderedDict
import sys
import random
import os
from sklearn.cluster import KMeans




class DataProcesserSeg:
    def __init__(self, archive_path, col_cloud='cloud_id', col_wave='wave_id', col_coor=['xcoordinate', 'ycoordinate', 'time_point'], groups=None, **kwargs):
        
        self.archive_path = archive_path
        self.archive = zipfile.ZipFile(self.archive_path, 'r')
        self.col_cloud = col_cloud
        self.col_coor = col_coor
        self.col_wave = col_wave

        self.dataset = None
    
        
        self.train_set = None
        self.validation_set = None
        self.test_set = None

        self.nclouds = None

        self.logs = []

        self.read_archive()

        self.groups = groups
        if self.groups is None:
            self.detect_groups()
        
        
        
            
        

    def read_archive(self):
        """
        Read a zip archive, without extraction, than contains:

        * data as .csv: cloud id, coordinates, and measurements in columns. Names of columns must have the format:
            FOV, x_coordinate, y_coordinate, time_point, measurement1, measurement2, ... , where the measurements can have any name
         
        * Annotations of waves in the form of (x_min, y_min, time_min, x_max, y_max, time_max)
            The dataframe must have the format of columns:
            FOV, x_min, y_min, time_min, x_max, y_max, time_max
        
        :return: 2 pandas, one with raw data, one with the box annotations
        """
        # import the dataset and the boxes
        self.dataset = pd.read_csv(self.archive.open('dataset.csv'))
        
        self.check_datasets()

        # create a list of all the cloud indexes
        self.nclouds = pd.unique(self.dataset[self.col_cloud]).tolist()

        # get the measurement groups
        groups = list(self.dataset.columns.values)
        groups.remove(self.col_cloud)
        for c in self.col_coor:
            groups.remove(c)
        self.groups = groups

        self.logs.append('Read archive: {0}'.format(self.archive_path))
        return None
    

    def split_data(self, ftrain=.6, fvalidate=.3, ftest=.1):
        """ 
        Split the dataset into train, validation, and test set based on the fractions

        :return: 3 pandas, one for each set
        """
        
        assert ftrain + fvalidate + ftest > 1-1e-4 and ftrain + fvalidate + ftest < 1+1e-4, 'The set split does not add to 100%. It sums up to: ' + str(ftrain + fvalidate + ftest)

        random.shuffle(self.nclouds)
        train_last = int((len(self.nclouds)+1)*ftrain)
        validation_last = train_last + int((len(self.nclouds)+1)*fvalidate)
                    
        train_id = list(self.nclouds[:train_last])
        validation_id = list(self.nclouds[train_last:validation_last])
        test_id = list(self.nclouds[validation_last:])

        self.train_set = self.dataset[self.dataset[self.col_cloud].isin(train_id)]

        self.validation_set = self.dataset[self.dataset[self.col_cloud].isin(validation_id)]
        
        self.test_set = self.dataset[self.dataset[self.col_cloud].isin(test_id)]
        
####TODO: PUT A CHECK TO SEE THAT NONE OF THE SUBSETS ARE EMPTY
        return None

    
    def check_datasets(self):
        """
        Check that there is at least one correct measurement in dataset, and values in set_ids.
        :return:
        """
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        self.dataset.columns = self.dataset.columns.str.lower()
        colnames_dataset = list(self.dataset.columns.values)
        
        colnames_dataset.remove(self.col_cloud)
        for c in self.col_coor:
            colnames_dataset.remove(c)

        # check the cloud id is present everywhere
        if not self.col_cloud in self.dataset.columns.values:
            warnings.warn('Field of View column "{}" is missing in dataset.csv.'.format(self.col_cloud))
        
        # check the coordinates is present
        for c in self.col_coor:
            if not c in self.dataset.columns.values:
                warnings.warn('Coordinate or time columns "{}" are missing in dataset.csv.'.format(c))
        
        # check if there is any data in the dataframe
        if self.dataset.select_dtypes(numerics).empty:
            warnings.warn('No data in dataset.csv.')
        
        # check if there are measurements
        if self.dataset[colnames_dataset].select_dtypes(numerics).empty:
            warnings.warn('No numerical measurement columns in dataset.csv.')
        
        # check if there are duplicated points
        #if any(self.dataset[self.col_coor].duplicated()):
        #    warnings.warn('Found duplicated point in dataset.csv.')
        # check if there are duplicated boxes
        
        return None
    


    def detect_groups(self):
        colnames = list(self.dataset.columns.values)
        colnames.remove(self.col_cloud)
        colnames.remove(self.col_wave)

        for c in self.col_coor:
            colnames.remove(c)
        groups = list(OrderedDict.fromkeys([i for i in colnames]))
        self.groups = groups

        return None
    


    


    


  










class CloudDatasetSeg():
    """Standard dataset object with ID, class"""

    def __init__(self, dataset, connection_method='knn', radius=2, col_cloud='cloud_id', col_coor=['xcoordinate', 'ycoordinate', 'time_point'], col_wave='wave_id', groups=None, n_clusters=2, method='center', norm_per_meas=False):
        """
        General Dataset class for arbitrary uni and multivariate point clouds.
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        # Read dataset (since the whole dataset is split in the read in, the dataset passed here is only either train, test, or validate)
        self.dataset = dataset

        self.col_cloud = col_cloud
        self.col_coor = col_coor
        self.col_wave = col_wave

        self.logs = []

        self.nclouds = pd.unique(self.dataset[self.col_cloud]).tolist()

        #self.col_cloud_idx = self.dataset.columns.get_loc(self.col_cloud)
        #self.col_coor_idx = self.dataset.columns.get_loc(self.col_coor)

        self.euler_stats = None
        self.dimension_stats = None
        self.kmeans_clusters = None
        self.n_clusters = n_clusters

        self.groups = groups
        if self.groups is None:
            self.detect_groups()

        #self.graph = None
        self.connection_method = connection_method.upper()
        assert self.connection_method in ['KNN', 'RADIUS'], 'The available methods to connect the points are \'knn\' and \'radius\''
        self.radius = radius

        
        self.normalize_meas(method=method, norm_per_meas=norm_per_meas)

    def __len__(self):
        """
        Returns the amount of clouds
        """
        return len(self.nclouds)

    def __getitem__(self, i):
        
        cloud = self.dataset[self.dataset[self.col_cloud] == self.nclouds[i]]
        print(self.nclouds[i])
        ## to graph
        # Node features
        node_features = torch.tensor(cloud[self.groups].values, dtype=torch.float)
        # Node positions
        node_positions = torch.tensor(cloud[self.col_coor].values, dtype=torch.float)
        # Node labels 
        node_labels = torch.tensor(cloud[self.col_wave].values, dtype=torch.float)

####### TODO: introduce more graph connection methods
        # Convert to point cloud (=localized graph without edges)
        #if self.connection_method == 'KNN':
        #    connect_transform = KNNGraph(k=self.radius)
        #if self.connection_method == 'RADIUS':
        #    connect_transform = RadiusGraph(r=self.radius)
        cloud = Data(x=node_features, pos=node_positions, y=node_labels)
        #cloud = connect_transform(cloud)

    
        
        #cloud, box = transform(cloud, box, axis=2, p=0.5)

        return {'x': cloud.x, 'pos': cloud.pos, 'y': cloud.y}

    
    
    def detect_groups(self):
        colnames = list(self.dataset.columns.values)
        colnames.remove(self.col_cloud)
        colnames.remove(self.col_wave)

        for c in self.col_coor:
            colnames.remove(c)
        groups = list(OrderedDict.fromkeys([i for i in colnames]))
        self.groups = groups

        return None
    

    def normalize_meas(self, method='zscore', norm_per_meas=False):
        """
        Normalize the measurements for each point
        :param: methods, 'center', 'zscore', 'minmaxscaling'
        :param: norm_per_meas, should the 
        :return: Modifies .dataset in-place and returns info about process (mean train...)
        """

        methods = ['center', 'zscore', 'minmaxscaling']

        assert method in methods, 'Invalid method, choose from: {0}'.format(methods)

##### TODO: this can be done directly in a pg transform thing (NormalizeFeatures)
        ## standardize per group
        if norm_per_meas:
            mu = {g:[] for g in self.groups}
            sd = {g:[] for g in self.groups}
            mini = {g:[] for g in self.groups}
            maxi = {g:[] for g in self.groups}

            for group in self.groups:
                group_data = self.dataset[group]
                mu[group] = np.nanmean(group_data)
                sd[group] = np.nanstd(group_data)
                mini[group] = np.nanmin(group_data)
                maxi[group] = np.nanmax(group_data)

                self.logs.append('Individual stats for group {}; mu:{}; sd:{}; minimum:{}; maximum:{}'.format(group, mu[group], sd[group], mini[group], maxi[group]))

                if method == 'center':
                    self.dataset[group] -= mu[group]
                if method == 'zscore':
                    self.dataset[group] = (self.dataset[group] - mu[group]) / sd[group]
                if method == 'minmaxscaling':
                    self.dataset[group] = (self.dataset[group] - mini[group]) / (maxi[group] - mini[group])

        ## standardize globally
        else:
            group_data = self.dataset[self.groups]
            mu = np.nanmean(group_data)
            sd = np.nanstd(group_data)
            mini = np.nanmin(group_data)
            maxi = np.nanmax(group_data)

            self.logs.append('Global stats; mu:{}; sd:{}; minimum:{}; maximum:{}'.format(mu, sd, mini, maxi))

            if method == 'center':
                self.dataset[self.groups] -= mu
            if method == 'zscore':
                self.dataset[self.groups] = (self.dataset[self.groups] - mu) / sd
            if method == 'minmaxscaling':
                self.dataset[self.groups] = (self.dataset[self.groups] - mini) / (maxi - mini)

        self.logs.append('Process: method:{}, norm_per_meas:{}'.format(method, norm_per_meas))

        return None