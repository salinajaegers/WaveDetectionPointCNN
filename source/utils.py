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


def transform(cloud, boxes, axis=2, p=0.5):
    """
    :param cloud: a point cloud as pytorch_geometric Data class
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: transformed cloud, transformed bounding box coordinates
    """
    mu_cloud = cloud.pos.mean(dim=-2, keepdim=True)
    scale = (1 / cloud.pos.abs().max()) * 0.999999

    boxes = (boxes - mu_cloud) * scale
    cloud.pos = (cloud.pos - mu_cloud) * scale

    if random.random() < p:
        pos = cloud.pos.clone()
        pos[:, axis] = -pos[:, axis]
        cloud.pos = pos
        boxes[:,axis] = -boxes[:,axis]
        boxes[:,3+axis] = -boxes[:,3+axis]

    return cloud, boxes
