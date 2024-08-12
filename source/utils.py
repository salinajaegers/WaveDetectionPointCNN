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

import json
import os
import torch
import random
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as FT



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


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    lower_bounds = torch.max(set_1[:, :3].unsqueeze(1), set_2[:, :3].unsqueeze(0))  # (n1, n2, 3)
    upper_bounds = torch.min(set_1[:, 3:].unsqueeze(1), set_2[:, 3:].unsqueeze(0))  # (n1, n2, 3)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 3)
    intersection = intersection_dims[:, :, 0] * intersection_dims[:, :, 1] * intersection_dims[:, :, 2] # (n1, n2)

    
    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 3] - set_1[:, 0]) * (set_1[:, 4] - set_1[:, 1]) * (set_1[:, 5] - set_1[:, 2]) # (n1)
    areas_set_2 = (set_2[:, 3] - set_2[:, 0]) * (set_2[:, 4] - set_2[:, 1]) * (set_2[:, 5] - set_2[:, 2])  # (n2)

    # Find the union
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)




