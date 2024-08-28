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
import random
import os
from sklearn.cluster import KMeans

from utils import transform

class DataProcesser:
    def __init__(self, archive_path, col_cloud='FOV', col_coor=['x_coordinate', 'y_coordinate', 'time_point'], col_box=['x_min', 'y_min', 'time_min', 'x_max', 'y_max', 'time_max'], groups=None, **kwargs):
        
        self.archive_path = archive_path
        self.archive = zipfile.ZipFile(self.archive_path, 'r')
        self.col_cloud = col_cloud
        self.col_coor = col_coor
        self.col_box = col_box

        self.dataset = None
        
        self.groups = groups
        if self.groups is None:
            self.detect_groups()
        
        self.train_set = None
        self.validation_set = None
        self.test_set = None

        self.nclouds = None

        self.logs = []

        self.read_archive()
        
        
        #self.split_data(ftrain, fvalidate, ftest)
            
        

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
        self.boxes = pd.read_csv(self.archive.open('boxes.csv'))
        self.check_datasets()

        # create a list of all the cloud indexes
        self.nclouds = pd.unique(self.dataset[self.col_cloud]).tolist()

        # get the measurement groups
        groups = list(self.dataset.columns.values)
        groups.remove(self.col_cloud)
        groups.remove(self.col_coor)
        self.groups = groups

        self.logs.append('Read archive: {0}'.format(self.archive_path))
        return None
    

    def split_data(self, ftrain=.65, fvalidate=.20, ftest=.15):
        """ 
        Split the dataset into train, validation, and test set based on the fractions

        :return: 3 pandas, one for each set
        """
        assert ftrain + fvalidate + ftest == 1, 'The set split does not add to 100%.'

        random.shuffle(self.nclouds)
        train_id = list(self.nclouds[:int((len(self.nclouds)+1)*ftrain)])
        validation_id = list(self.nclouds[int((len(self.nclouds)+1)*ftrain):int((len(self.nclouds)+1)*fvalidate)])
        test_id = list(self.nclouds[int((len(self.nclouds)+1)*fvalidate):])

        self.train_set = self.dataset[self.dataset[self.col_cloud].isin(train_id)]
        self.train_boxes = self.boxes[self.boxes[self.col_cloud].isin(train_id)]

        self.validation_set = self.dataset[self.dataset[self.col_cloud].isin(validation_id)]
        self.validation_boxes = self.boxes[self.boxes[self.col_cloud].isin(validation_id)]

        self.test_set = self.dataset[self.dataset[self.col_cloud].isin(test_id)]
        self.test_boxes = self.boxes[self.boxes[self.col_cloud].isin(test_id)]

####TODO: PUT A CHECK TO SEE THAT NONE OF THE SUBSETS ARE EMPTY
        return None

    
    def check_datasets(self):
        """
        Check that there is at least one correct measurement in dataset, and values in set_ids.
        :return:
        """
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        self.dataset.columns = self.dataset.columns.lower()
        self.boxes.columns = self.boxes.columns.lower()
        colnames_dataset = list(self.dataset.columns.values)
        colnames_dataset.remove(self.col_cloud)
        colnames_dataset.remove(self.col_coor)

        # check the cloud id is present everywhere
        if not self.col_cloud in self.dataset.columns.values:
            warnings.warn('Field of View column "{}" is missing in dataset.csv.'.format(self.col_cloud))
        if not self.col_cloud in self.boxes.columns.values:
            warnings.warn('Field of View column "{}" is missing in boxes.csv.'.format(self.col_cloud))
        # check the coordinates is present
        if not self.col_coor in self.dataset.columns.values:
            warnings.warn('Coordinate and time columns "{}" are missing in dataset.csv.'.format(self.col_coor))
        if not self.col_coor in self.boxes.columns.values:
            warnings.warn('Coordinate and time columns "{}" are missing in boxes.csv.'.format(self.col_coor))
        
        # check if there is any data in the dataframe
        if self.dataset.select_dtypes(numerics).empty:
            warnings.warn('No data in dataset.csv.')
        # check if there is any data in the boxes
        if self.boxes.select_dtypes(numerics).empty:
            warnings.warn('No numerical columns in boxes.csv.')
        # check if there are measurements
        if self.dataset[colnames_dataset].select_dtypes(numerics).empty:
            warnings.warn('No numerical measurement columns in dataset.csv.')
        
        # check if there are duplicated points
        if any(self.dataset[self.col_coor].duplicated()):
            warnings.warn('Found duplicated point in dataset.csv.')
        # check if there are duplicated boxes
        if any(self.boxes[self.col_box].duplicated()):
            warnings.warn('Found duplicated box in boxes.csv.')

        return None
    


    def detect_groups(self):
        colnames = list(self.dataset.columns.values)
        colnames.remove(self.col_id)
        colnames.remove(self.col_class)
        groups = list(OrderedDict.fromkeys([i.split('_')[0] for i in colnames]))
        self.groups = groups

        return None
    


    


    


  










class CloudDataset():
    """Standard dataset object with ID, class"""

    def __init__(self, dataset, boxes, connection_method='knn', radius=2, col_cloud='FOV', col_coor=['x_coordinate', 'y_coordinate', 'time_point'], col_box=['width','length','height','center_x','center_y','center_z','euler_z','euler_y','euler_x'], groups=None, n_clusters=3):
        """
        General Dataset class for arbitrary uni and multivariate point clouds.
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        # Read dataset (since the whole dataset is split in the read in, the dataset passed here is only either train, test, or validate)
        self.dataset = dataset
        self.boxes = boxes

        self.col_cloud = col_cloud
        self.col_coor = col_coor
        self.col_box = col_box

        self.nclouds = pd.unique(self.dataset[self.col_cloud]).tolist()

        self.col_cloud_idx = self.dataset.columns.get_loc(self.col_cloud)
        self.col_coor_idx = self.dataset.columns.get_loc(self.col_coor)

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

        self.groups_indices = {}
        self.get_groups_indices()
        self.binnin_stats()

    def __len__(self):
        """
        Returns the amount of clouds
        """
        return len(self.nclouds)

    def __getitem__(self, i):
        # Read image
        cloud = self.dataset[self.dataset[self.col_cloud] == self.nclouds[i]]
        ## to graph
        # Node features
        node_features = torch.tensor(cloud[self.groups].values, dtype=torch.float)
        # Node positions
        node_positions = torch.tensor(cloud[self.col_coor].values, dtype=torch.float)

####### TODO: introduce more graph connection methods
        # Convert to point cloud (=localized graph without edges)
        #if self.connection_method == 'KNN':
        #    connect_transform = KNNGraph(k=self.radius)
        #if self.connection_method == 'RADIUS':
        #    connect_transform = RadiusGraph(r=self.radius)
        cloud = Data(x=node_features, pos=node_positions)
        #cloud = connect_transform(cloud)

        # Read objects in this image (bounding boxes, labels, difficulties)
        boxes = self.boxes[self.boxes[self.col_cloud] == self.nclouds[i]][self.col_box]
        boxes = boxes[self.col_box]
        boxes = torch.FloatTensor(boxes)  # (n_objects, 9)
#### idk if this whole label thing works
        #cloud, boxes = transform(cloud, boxes, axis=2, p=0.5)

        

        return {'x': cloud.x, 'pos': cloud.pos,'boxes': boxes}
    
    def detect_groups(self):
        colnames = list(self.dataset.columns.values)
        colnames.remove(self.col_id)
        colnames.remove(self.col_class)
        groups = list(OrderedDict.fromkeys([i.split('_')[0] for i in colnames]))
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

            self.logs.append('Global stats; mu:{}; sd:{}; minimum:{}; maximum:{}'.format(group, mu, sd, mini, maxi))

            if method == 'center':
                self.dataset[self.groups] -= mu
            if method == 'zscore':
                self.dataset[self.groups] = (self.dataset[self.groups] - mu) / sd
            if method == 'minmaxscaling':
                self.dataset[self.groups] = (self.dataset[self.groups] - mini) / (maxi - mini)

        self.logs.append('Process: method:{}, norm_per_meas:{}'.format(method, norm_per_meas))

        return None
    

    def binnin_stats(self):
        """
        The stats used to build the bins used to create priors.
        """
        # euler stats
        euler_x = self.boxes['euler_x']
        euler_y = self.boxes['euler_y']
        euler_z = self.boxes['euler_z']

        mu_x = np.nanmean(euler_x)
        mu_y = np.nanmean(euler_y)
        mu_z = np.nanmean(euler_z)

        sd_x = np.nanstd(euler_x)
        sd_y = np.nanstd(euler_y)
        sd_z = np.nanstd(euler_z)

        min_x, max_x = np.nanmin(euler_x), np.nanmax(euler_x)
        min_y, max_y = np.nanmin(euler_y), np.nanmax(euler_y)
        min_z, max_z = np.nanmin(euler_z), np.nanmax(euler_z)

        self.euler_stats = {'mu': [mu_x, mu_y, mu_z], 'sd': [sd_x, sd_y, sd_z], 'min': [min_x, min_y, min_z], 'max': [max_x, max_y, max_z]}

        # dimension stats
        width = self.boxes['width']
        length = self.boxes['length']
        height = self.boxes['height']

        mu_x = np.nanmean(width)
        mu_y = np.nanmean(length)
        mu_z = np.nanmean(height)

        sd_x = np.nanstd(width)
        sd_y = np.nanstd(length)
        sd_z = np.nanstd(height)

        min_x, max_x = np.nanmin(width), np.nanmax(width)
        min_y, max_y = np.nanmin(length), np.nanmax(length)
        min_z, max_z = np.nanmin(height), np.nanmax(height)

        self.dimension_stats = {'mu': [mu_x, mu_y, mu_z], 'sd': [sd_x, sd_y, sd_z], 'min': [min_x, min_y, min_z], 'max': [max_x, max_y, max_z]}


        # the k-means detected centers
        kmeans_width = KMeans(n_clusters=self.n_clusters, random_state=7, n_init="auto").fit(width).cluster_centers_
        kmeans_length = KMeans(n_clusters=self.n_clusters, random_state=7, n_init="auto").fit(length).cluster_centers_
        kmeans_height = KMeans(n_clusters=self.n_clusters, random_state=7, n_init="auto").fit(height).cluster_centers_

        kmeans_eulerz = KMeans(n_clusters=self.n_clusters, random_state=7, n_init="auto").fit(euler_z).cluster_centers_
        kmeans_eulery = KMeans(n_clusters=self.n_clusters, random_state=7, n_init="auto").fit(euler_y).cluster_centers_
        kmeans_eulerx = KMeans(n_clusters=self.n_clusters, random_state=7, n_init="auto").fit(euler_x).cluster_centers_

        self.kmeans_clusters = {'width': kmeans_width, 'length': kmeans_length, 'height': kmeans_height, 'euler_z': kmeans_eulerz, 'euler_y': kmeans_eulery, 'euler_x': kmeans_eulerx}

        return None
    
#    def graph_processing(self):
#        """
#        Process data for neural network.
#        :return: Modifies .dataset in-place and returns a graph representation of it
#        """
#        ## to tensor 
#        ls_transforms = transforms.Compose([
#        RandomCrop(output_size=length, ignore_na_tails=True),
#        Subtract(average_perChannel),
#        ToTensor()])
#
#        ## to graph
#        # Node features
#        node_features = torch.tensor(self.dataset[self.groups].values, dtype=torch.float)
#        # Node positions
#        node_positions = torch.tensor(self.dataset[self.col_coor].values, dtype=torch.float)
#
######## TODO: introduce more graph connection methods
#        # Convert to point cloud (=localized graph without edges)
#        connect_transform = KNNGraph(k=self.knnradius)
#        self.graph = Data(x=node_features, pos=node_positions)
#        self.graph = connect_transform(self.graph)
#        return None



