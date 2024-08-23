import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    MLP,
    PointTransformerConv,
    fps,
    global_mean_pool,
    knn,
    knn_graph,
    knn_interpolate
)
from torch_geometric.utils import scatter
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import torchvision


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from itertools import product as product

from utils import *


# Set CPU/CUDA device based on availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############ FROM THE PYTORCH GEOMETRIC EXAMPLES #############################################
# point transformer help functions from pytorch geometric example
class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels)
        self.lin_out = Lin(out_channels, out_channels)

        self.pos_nn = MLP([3, 64, out_channels], norm=None, plain_last=False)

        self.attn_nn = MLP([out_channels, 64, out_channels], norm=None,
                           plain_last=False)

        self.transformer = PointTransformerConv(in_channels, out_channels,
                                                pos_nn=self.pos_nn,
                                                attn_nn=self.attn_nn)

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x

class TransitionDown(torch.nn.Module):
    """Samples the input point cloud by a ratio percentage to reduce
    cardinality and uses an mlp to augment features dimensionnality.



    REDUCE POINTS IN THE GRAPH TO A CERTAIN RATIO GIVEN BY SELF.RATIO.
    SO IF WE START WITH 10 POINTS WE REDUCE IT TO 5 IF WE CHOOSE RATIO=0.5
    """
    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False)

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch,
                            batch_y=sub_batch)

        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out = scatter(x[id_k_neighbor[1]], id_k_neighbor[0], dim=0,
                        dim_size=id_clusters.size(0), reduce='max')

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch
    
class TransitionUp(torch.nn.Module):
    """
    Reduce features dimensionality and interpolate back to higher
    resolution and cardinality.


    INTERPOLATE THE REDUCED NODE FEATURES BACK TO THE PREVIOUSLY POOLED NODES.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp_sub = MLP([in_channels, out_channels], plain_last=False)
        self.mlp = MLP([out_channels, out_channels], plain_last=False)

    def forward(self, x, x_sub, pos, pos_sub, batch=None, batch_sub=None):
        # transform low-res features and reduce the number of features
        x_sub = self.mlp_sub(x_sub)

        # interpolate low-res feats to high-res points
        x_interpolated = knn_interpolate(x_sub, pos_sub, pos, k=3,
                                         batch_x=batch_sub, batch_y=batch)

        x = self.mlp(x) + x_interpolated

        return x
    














################################ MAIN HELPER MODELS #############################################

# base convolution model
class BaseModel(torch.nn.Module):
    def __init__(self, nmeasurements, k=16):
        super().__init__()
        self.k = k

        # initial block
# not sure about the input for the mlp
        self.mlp_input = MLP([nmeasurements, 32], plain_last=False)
        self.transformer_input = TransformerBlock(in_channels=32,
                                                  out_channels=32)

        # backbone layers
        self.transformers_down = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        dim_model_down = [32, 64, 128, 256, 256, 512, 1024]
        
        for i in range(0, len(dim_model_down) - 1):
            # Add Transition Down block followed by a Point Transformer block
            self.transition_down.append(
                TransitionDown(in_channels=dim_model_down[i],
                               out_channels=dim_model_down[i + 1], k=self.k))

            self.transformers_down.append(
                TransformerBlock(in_channels=dim_model_down[i + 1],
                                 out_channels=dim_model_down[i + 1]))




    def forward(self, x, pos, batch=None):

        # add dummy features in case there is none
        if x is None:
            x = torch.ones((pos.shape[0], 1), device=pos.get_device())

        out_x = []
        out_pos = []
        out_batch = []

        # first block
        x = self.mlp_input(x)
##### WHY DO WE REDO THE EDGE INDEX??????
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # save outputs for skipping connections
        out_x.append(x)
        out_pos.append(pos)
        out_batch.append(batch)


# backbone down : #reduce cardinality and augment dimensionnality
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)
            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

            out_x.append(x)
            out_pos.append(pos)
            out_batch.append(batch)
          

        return out_x, out_pos, out_batch
    




# prediction convolution model
class PredictionModel(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.

    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 8732 prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.

    The class scores represent the scores of each object class in each of the 8732 bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes):
        """
        :param n_classes: number of different types of objects
        """
        super().__init__()

        self.n_classes = n_classes

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {'transform_4_feats': 4,
                   'transform_5_feats': 6,
                   'transform_6_feats': 6,
                   'transform_7_feats': 6}
        
        box_dim = 9
        # 4 prior-boxes implies we use 4 different aspect ratios, etc.


        #dim_model_down = [32, 64, 128, 128, 256, 512, 1024]
        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        #MLP([in_channels, out_channels], plain_last=False)
        self.loc_mlp4 = MLP([128, n_boxes['transform_4_feats'] * box_dim], plain_last=False)
        self.loc_mlp5 = MLP([256, n_boxes['transform_5_feats'] * box_dim], plain_last=False)
        self.loc_mlp6 = MLP([512, n_boxes['transform_6_feats'] * box_dim], plain_last=False)
        self.loc_mlp7 = MLP([1024, n_boxes['transform_7_feats'] * box_dim], plain_last=False)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_mlp4 = MLP([128, n_boxes['transform_4_feats'] * n_classes], plain_last=False)
        self.cl_mlp5 = MLP([256, n_boxes['transform_5_feats'] * n_classes], plain_last=False)
        self.cl_mlp6 = MLP([512, n_boxes['transform_6_feats'] * n_classes], plain_last=False)
        self.cl_mlp7 = MLP([1024, n_boxes['transform_7_feats'] * n_classes], plain_last=False)


        # Initialize convolutions' parameters
        self.init_param()

    def init_param(self):
        """
        Initialize convolution parameters from model applied before this.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, out_x, out_pos, out_batch):
        """
        Forward propagation.

        :param conv4_3_feats: conv4_3 feature map, a tensor of dimensions (N, 512, 38, 38)
        :param conv7_feats: conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :param conv8_2_feats: conv8_2 feature map, a tensor of dimensions (N, 512, 10, 10)
        :param conv9_2_feats: conv9_2 feature map, a tensor of dimensions (N, 256, 5, 5)
        :param conv10_2_feats: conv10_2 feature map, a tensor of dimensions (N, 256, 3, 3)
        :param conv11_2_feats: conv11_2 feature map, a tensor of dimensions (N, 256, 1, 1)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        batch_size = transform_4_feats.size(0)

##### TODO: check if the order of the prior box coordinates is right
### This might be different dimension wise since the point clouds do not have the same amount of points 
### We will see if this has an effect tho (hope not)
        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        print('THE SHAPE OF THE TRANSFOMS ARE: ' + str(transform_4_feats.shape))
        l_mlp4 = self.loc_mlp4(transform_4_feats)  # (N, 16, 38, 38)
        print('THE SHAPE OF THE TRANSFOMS ARE: ' + str(l_mlp4.shape))
        l_mlp4 = l_mlp4.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 16), to match prior-box order (after .view())
        # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
        l_mlp4 = l_mlp4.view(batch_size, -1, 4)  # (N, 5776, 4), there are a total 5776 boxes on this feature map

        l_mlp5 = self.loc_mlp5(transform_5_feats)  # (N, 24, 19, 19)
        print('THE SHAPE OF THE TRANSFOMS ARE: ' + str(l_mlp5.shape))
        l_mlp5 = l_mlp5.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 24)
        l_mlp5 = l_mlp5.view(batch_size, -1, 4)  # (N, 2166, 4), there are a total 2116 boxes on this feature map

        l_mlp6 = self.loc_mlp6(transform_6_feats)  # (N, 24, 10, 10)
        l_mlp6 = l_mlp6.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 24)
        l_mlp6 = l_mlp6.view(batch_size, -1, 4)  # (N, 600, 4)

        l_mlp7 = self.loc_mlp7(transform_7_feats)  # (N, 24, 5, 5)
        l_mlp7 = l_mlp7.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 24)
        l_mlp7 = l_mlp7.view(batch_size, -1, 4)  # (N, 150, 4)





        # Predict classes in localization boxes
        c_mlp4 = self.cl_mlp4(transform_4_feats)  # (N, 4 * n_classes, 38, 38)
        c_mlp4 = c_mlp4.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 4 * n_classes), to match prior-box order (after .view())
        c_mlp4 = c_mlp4.view(batch_size, -1, self.n_classes)  # (N, 5776, n_classes), there are a total 5776 boxes on this feature map

        c_mlp5 = self.cl_mlp5(transform_5_feats)  # (N, 6 * n_classes, 19, 19)
        c_mlp5 = c_mlp5.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 6 * n_classes)
        c_mlp5 = c_mlp5.view(batch_size, -1, self.n_classes)  # (N, 2166, n_classes), there are a total 2116 boxes on this feature map

        c_mlp6 = self.cl_mlp6(transform_6_feats)  # (N, 6 * n_classes, 10, 10)
        c_mlp6 = c_mlp6.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 6 * n_classes)
        c_mlp6 = c_mlp6.view(batch_size, -1, self.n_classes)  # (N, 600, n_classes)

        c_mlp7 = self.cl_mlp7(transform_7_feats)  # (N, 6 * n_classes, 5, 5)
        c_mlp7 = c_mlp7.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 6 * n_classes)
        c_mlp7 = c_mlp7.view(batch_size, -1, self.n_classes)  # (N, 150, n_classes)


        # A total of 8732 boxes
        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat([l_mlp4, l_mlp5, l_mlp6, l_mlp7], dim=1)  # (N, 8732, 4)
        classes_scores = torch.cat([c_mlp4, c_mlp5, c_mlp6, c_mlp7], dim=1)  # (N, 8732, n_classes)

        return locs, classes_scores



################################ LOSS MODEL #####################################################
# loss for box and class
class BoxLoss(nn.Module):
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super().__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()  # *smooth* L1 loss in the paper; see Remarks section in the tutorial
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)


    def forward(self, predicted_boxes, predicted_scores, boxes, labels):


        box_loss = self.smooth_l1(predicted_boxes[positive_priors], true_locs[positive_priors])
        cls_loss = self.cross_entropy(predicted_scores.view(-1, nclasses), true_classes.view(-1))

        return cls_loss + self.alpha * box_loss






################################ MAIN MODEL #####################################################
# put all previous models together in a network
class WavePrediction(pl.LightningModule):
    """
    The main network - encapsulates the base network and prediction convolutions.
    """

    def __init__(self, nclasses, bins, feat_pos, batch_size, lr_scheduler_milestones, lr_gamma, nfeatures, length, nmeasurement, lr=1e-2, L2_reg=1e-3, top_acc=1, loss=BoxLoss(), accuracy=F1_acc()):
        super().__init__()

        self.batch_size = batch_size
        self.nclasses = nclasses
        self.nfeatures = nfeatures
        self.length = length
        self.nmeasurement = nmeasurement

        self.loss = loss
        self.lr = lr
        self.lr_scheduler_milestones = lr_scheduler_milestones
        self.lr_gamma = lr_gamma
        self.L2_reg = L2_reg

        
        self.bins = bins
        self.feat_pos = feat_pos

        # Log hyperparams (all arguments are logged by default)
        self.save_hyperparameters(
            'length',
            'nfeatures',
            'L2_reg',
            'lr',
            'lr_gamma',
            'lr_scheduler_milestones',
            'batch_size',
            'nclasses'
        )

        # Metrics to log
        if not top_acc < nclasses:
            raise ValueError('`top_acc` must be strictly smaller than `nclasses`.')
        self.train_acc = accuracy()
        self.val_acc = accuracy()

        

        self.base = BaseModel()
        self.pred_convs = PredictionModel(nclasses)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, x, pos, batch=None):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network convolutions (lower level feature map generators)
        out_x, out_pos, out_batch = self.base(x, pos, batch)  # (N, 512, 38, 38), (N, 1024, 19, 19)

        
        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        boxes, scores = self.pred_convs(out_x, out_pos, out_batch)

        return boxes, scores



##### THE LIGHTNING FUNCTIONS
    @property
    def input_size(self):
        # Add a dummy channel dimension for conv1D layer
        return (self.batch_size, 1, self.length)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.L2_reg)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_scheduler_milestones, gamma=self.lr_gamma),
            'name': 'LearningRate'
        }
        return [optimizer], [lr_scheduler]

    def on_train_start(self):
        # Add mean accuracies per epoch to the hyperparam Tensorboard's tab
        self.logger.log_hyperparams(self.hparams, {'hp/train_acc': 0, 'hp/val_acc': 0})

    def training_step(self, batch, batch_idx):
        x, pos, box = batch['x'], batch['pos'], batch['box']
        predicted_boxes, predicted_scores = self(x, pos) 
        train_loss = self.loss(predicted_boxes, predicted_scores, box)
##### CHANGE PREDICTION TO THE BOXES AND LABELS THAT HAVE BEEN PREDICTED
        train_acc = self.train_acc(F.softmax(prediction, dim=1), label)
        
        # Add logs of the batch to tensorboard
        self.log('train_loss', train_loss, on_step=True)
        self.log('train_acc', train_acc, on_step=True)
        return {'loss': train_loss, 'prediction_boxes': predicted_boxes, 'target_boxes': boxes, 'prediction_labels': predicted_boxes, 'target_labels': boxes}

    # This hook receive the outputs of all training steps as a list of dictionaries
    def training_epoch_end(self, train_outputs):
        mean_loss = torch.stack([k['loss'] for k in train_outputs]).mean()
        train_acc = self.train_acc.compute()
        
        self.log('MeanEpoch/train_loss', mean_loss)
        # The .compute() of Torchmetrics objects compute the average of the epoch and reset for next one
        self.log('MeanEpoch/train_acc', train_acc)
        self.log('hp/train_acc', train_acc)

    def validation_step(self, batch, batch_idx):
        x, pos, box = batch['x'], batch['pos'], batch['box']
        predicted_boxes, predicted_scores = self(x, pos) 
        val_loss = self.loss(predicted_boxes, predicted_scores, box)
##### CHANGE PREDICTION TO THE BOXES AND LABELS THAT HAVE BEEN PREDICTED
        val_acc = self.val_acc(F.softmax(prediction, dim=1), label)

        self.log('MeanEpoch/val_acc', val_acc, on_epoch=True, prog_bar=True)
        self.log('hp/val_acc', val_acc)
        return {'loss': val_loss, 'prediction_boxes': predicted_boxes, 'target_boxes': boxes, 'prediction_labels': predicted_boxes, 'target_labels': boxes}

    # This hook receive the outputs of all validation steps as a list of dictionaries
    def validation_epoch_end(self, val_outputs):
        mean_loss = torch.stack([x['loss'] for x in val_outputs]).mean()
        # Create a figure of the confmat that is loggable
        preds = torch.cat([F.softmax(x['preds'], dim=1) for x in val_outputs])
        target = torch.cat([x['target'] for x in val_outputs])
        confmat = torchmetrics.functional.confusion_matrix(preds, target, normalize=None, num_classes=self.nclasses)
        df_confmat = pd.DataFrame(confmat.cpu().numpy(), index = range(self.nclasses), columns = range(self.nclasses))
        plt.figure(figsize = (10,7))
        fig_ = sns.heatmap(df_confmat, annot=True, cmap='Blues').get_figure()
        plt.close(fig_)

        self.log('MeanEpoch/val_loss', mean_loss)
        self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items




##### CUSTOM CLASS FUNCTIONS
    def create_prior_boxes(self):
        """
        Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.

        :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
        """

        # instead of the fmap_dims create a fake grid or try to make a prior for each point that is left over after the tranformations
        fmap_dims = {'transform_4': self.feat_pos[0],
                     'transform_5': self.feat_pos[1],
                     'transform_6': self.feat_pos[2],
                     'transform_7': self.feat_pos[3]}

        obj_scales = {'transform_4': 0.1,
                      'transform_5': 0.3,
                      'transform_6': 0.6,
                      'transform_7': 0.9}

        aspect_ratios = {'transform_4': [1., 2., 0.5],
                         'transform_5': [1., 2., 3., 0.5, .333],
                         'transform_6': [1., 2., 3., 0.5, .333],
                         'transform_7': [1., 2., 3., 0.5, .333]}
        
        euler_angles = {'transform_4': self.bins[0],
                        'transform_5': self.bins[1],
                        'transform_6': self.bins[2],
                        'transform_7': self.bins[3]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for p in fmap_dims[fmap]:
                cx, cy, cz = p

                for e in euler_angles[fmap]: 
                    ez, ey, ex = e

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio), obj_scales[fmap] * sqrt(ratio), cx, cy, cz, ez, ey, ex])


        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
        prior_boxes.clamp_(0, 1)  # (8732, 4); this line has no effect; see Remarks section in tutorial

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_clouds_boxes = list()
        all_clouds_labels = list()
        all_clouds_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to

##### HOW TO DECODE OUR BOXES? ALSO THERE IS A LOG TRANSFORM SOMEWHERE IN THESE HELPER FUNCTIONS, FIND OUT WHAT IT DO BE DOIN
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            cloud_boxes = list()
            cloud_labels = list()
            cloud_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Check for each class
            for c in range(1, self.nclasses):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                cloud_boxes.append(class_decoded_locs[1 - suppress])
                cloud_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                cloud_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(cloud_boxes) == 0:
                cloud_boxes.append(torch.FloatTensor([[1., 1., 1., 0., 0., 0., 0., 0., 0.]]).to(device))
                cloud_labels.append(torch.LongTensor([0]).to(device))
                cloud_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            cloud_boxes = torch.cat(cloud_boxes, dim=0)  # (n_objects, 4)
            cloud_labels = torch.cat(cloud_labels, dim=0)  # (n_objects)
            cloud_scores = torch.cat(cloud_scores, dim=0)  # (n_objects)
            n_objects = cloud_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                cloud_scores, sort_ind = cloud_scores.sort(dim=0, descending=True)
                cloud_scores = cloud_scores[:top_k]  # (top_k)
                cloud_boxes = cloud_boxes[sort_ind][:top_k]  # (top_k, 4)
                cloud_labels = cloud_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_clouds_boxes.append(cloud_boxes)
            all_clouds_labels.append(cloud_labels)
            all_clouds_scores.append(cloud_scores)

        return all_clouds_boxes, all_clouds_labels, all_clouds_scores  # lists of length batch_size
