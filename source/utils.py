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

from scipy.spatial.transform import Rotation
from pytorch3d.ops import box3d_overlap

import numpy.linalg as LA
import matplotlib.pyplot as plt
#intersection_vol, iou_3d = box3d_overlap(boxes1, boxes2)


# Set the CPU/GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label map
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}


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

def find_intersection(box1, box2):
    """
    FROM THE box3d_overlap IMPLEMENTATION: 
    Inputs boxes1, boxes2 are tensors of shape (B, 8, 3)
    (where B doesn't have to be the same for boxes1 and boxes2),
    containing the 8 corners of the boxes, as follows:

        (4) +---------+. (5)
            | ` .     |  ` .
            | (0) +---+-----+ (1)
            |     |   |     |
        (7) +-----+---+. (6)|
            ` .   |     ` . |
            (3) ` +---------+ (2)


    NOTE: Throughout this implementation, we assume that boxes
    are defined by their 8 corners exactly in the order specified in the
    diagram above for the function to give correct results. In addition
    the vertices on each plane must be coplanar.
    As an alternative to the diagram, this is a unit bounding
    box which has the correct vertex ordering:

    box_corner_vertices = [
        [0, 0, 0],          xmin, ymin, zmin
        [1, 0, 0],          xmax, ymin, zmin
        [1, 1, 0],          xmax, ymax, zmin
        [0, 1, 0],          xmin, ymax, zmin
        [0, 0, 1],          xmin, ymin, zmax
        [1, 0, 1],          xmax, ymin, zmax
        [1, 1, 1],          xmax, ymax, zmax
        [0, 1, 1],          xmin, ymax, zmax
    ]

    Args:
        boxes1: tensor of shape (N, 8, 3) of the coordinates of the 1st boxes
        boxes2: tensor of shape (M, 8, 3) of the coordinates of the 2nd boxes
    Returns:
        vol: (N, M) tensor of the volume of the intersecting convex shapes
        iou: (N, M) tensor of the intersection over union which is
            defined as: `iou = vol / (vol1 + vol2 - vol)`
    """
    
    
    intersection_vol, iou_3d = box3d_overlap(box1, box2)



def F1_acc(predicted_boxes, predicted_scores, boxes, threshold=0.5):

    assert len(predicted_boxes) == len(predicted_scores) == len(boxes)
    
    # predicted_boxes dim: (N, n_priors, 9)
    # predicted_scores dim: (N, n_priors)

    # boxes dim: (n_realboxes, 9)
    boxes = torch.cat(boxes, dim=0)  # (n_objects, 24)
    boxes = boxes.view(-1, 8, 3)
    predicted_boxes = torch.cat(predicted_boxes, dim=0)  # (n_objects, 24)
    predicted_boxes = predicted_boxes.view(-1, 8, 3)
    predicted_scores = torch.cat(predicted_scores, dim=0)  # (n_objects)

    n_priors = predicted_scores.size(0)
    n_boxes = boxes.size(0)
    true_positives = torch.zeros((n_priors), dtype=torch.float).to(device)  # (n_class_detections)
    false_positives = torch.zeros((n_priors), dtype=torch.float).to(device)  # (n_class_detections)
    

    intersection_vol, iou_3d = box3d_overlap(predicted_boxes, boxes)  #iou: (N_predbox, N_truebox)

    
    for p in range(n_priors):
    # True Positives
        if torch.sum((iou_3d[p,:]>threshold).float()) == 1:
            true_positives[p] = 1
    # False Positives
        if torch.sum((iou_3d[p,:]>threshold).float()) > 1:
            false_positives[p] = 1

    n_true_positives = torch.cumsum(true_positives, dim=0)  
    n_false_positives = torch.cumsum(false_positives, dim=0) 

    #Precision = (True Positive)/(True Positive + False Positive)
    precision = n_true_positives / (
        n_true_positives + n_false_positives + 1e-10)


    #Recall = (True Positive)/(True Positive + False Negative)
    # we cant have and false negatives so that is =0
    recall = n_true_positives / (n_true_positives + 1e-10)

    #F1 score = (Precision × Recall)/[(Precision + Recall)/2]
    f1_score = (precision * recall) / ((precision + recall) / 2)
    

    return f1_score

def rotation_to_vertex(box):
    """
    Convert a box in format (width, lenght, height, [center], [euler angles])
    to the format of indicating all the vertices for pytorch3d iou computation.
    :param: box tensor(1,9): a box in format dimensions, center, euler angles
    
    

    box_corner_vertices = [
        [0, 0, 0],          xmin, ymin, zmin
        [1, 0, 0],          xmax, ymin, zmin
        [1, 1, 0],          xmax, ymax, zmin
        [0, 1, 0],          xmin, ymax, zmin
        [0, 0, 1],          xmin, ymin, zmax
        [1, 0, 1],          xmax, ymin, zmax
        [1, 1, 1],          xmax, ymax, zmax
        [0, 1, 1],          xmin, ymax, zmax
    ]
    """
#### TODO: maybe the tensor and the numpy will be needed to make uniformly
    w, l, h = box[:3]
    print(w, l, h)
    cx, cy, cz = box[3:6]
    euler_z, euler_y, euler_x = box[6:]
    
    # Local vertices centered at the origin
    local_vertices = torch.tensor([
        [-w/2, -l/2, -h/2],
        [w/2, -l/2, -h/2],
        [w/2, l/2, -h/2],
        [-w/2, l/2, -h/2],
        [-w/2, -l/2, h/2],
        [w/2, -l/2, h/2],
        [w/2, l/2, h/2],
        [-w/2, l/2, h/2]
    ])
    
    # Rotation matrix
    r = Rotation.from_euler('ZYX', [euler_z, euler_y, euler_x], degrees=True).as_matrix()
    r = torch.from_numpy(r)
    print(r)
    
    # Rotate and translate vertices
    rotated_vertices = np.dot(local_vertices, r.T)
    final_vertices = torch.from_numpy(rotated_vertices) + torch.tensor([cx, cy, cz])
    
    return final_vertices









def vertex_to_rotation(box):
    """
    Convert a box in format (width, lenght, height, [center], [euler angles])
    to the format of indicating all the vertices for pytorch3d iou computation.
    :param: box tensor(1,24): a box in format dimensions, center, euler angles
    
    

    box_corner_vertices = [
        [0, 0, 0],          xmin, ymin, zmin
        [1, 0, 0],          xmax, ymin, zmin
        [1, 1, 0],          xmax, ymax, zmin
        [0, 1, 0],          xmin, ymax, zmin
        [0, 0, 1],          xmin, ymin, zmax
        [1, 0, 1],          xmax, ymin, zmax
        [1, 1, 1],          xmax, ymax, zmax
        [0, 1, 1],          xmin, ymax, zmax
    ]
    """
    box = box.view(8,3)

    xmin, xmax = torch.min(box[:,0]), torch.max(box[:,0])
    ymin, ymax = torch.min(box[:,1]), torch.max(box[:,1])
    zmin, zmax = torch.min(box[:,2]), torch.max(box[:,2])

    center = np.array([(xmin + xmax)/2, (ymin + ymax)/2, (zmin + zmax)/2])

    centered_box = box - center
    
    edges = np.array([
        centered_box[1] - centered_box[0],  # Edge along the width
        centered_box[3] - centered_box[0],  # Edge along the length
        centered_box[4] - centered_box[0]   # Edge along the height
    ])
    
    width = np.linalg.norm(edges[0])
    length = np.linalg.norm(edges[1])
    height = np.linalg.norm(edges[2])
    
    # Normalize edges to get the rotation matrix (relative to the standard axes)
    r = np.column_stack([edges[0]/width, edges[1]/length, edges[2]/height])

    # Ensure the rotation matrix is orthogonal
    u, _, vh = np.linalg.svd(r)
    r = np.dot(u, vh)

    # Convert the rotation matrix to Euler angles
    r = Rotation.from_matrix(r)
    euler_angles = r.as_euler('ZYX', degrees=True)
    
    
    box = torch.tensor([width, length, height, center[0], center[1], center[2], euler_angles[0], euler_angles[1], euler_angles[2]])
    
    return box


