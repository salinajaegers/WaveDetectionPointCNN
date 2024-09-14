import os.path as osp

import torch
import torch.nn.functional as F
from torchmetrics.functional import jaccard_index
import pytorch_lightning as pl

import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader

from torch.nn import Linear as Lin

import torch_geometric.transforms as T
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
import sys

path_to_module = './source2'  # Path where all the .py files are, relative to the notebook folder
sys.path.append(path_to_module)
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    """Reduce features dimensionality and interpolate back to higher
    resolution and cardinality.
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



################################ LOSS MODEL #####################################################
# loss for instance and class 
class SegLoss(nn.Module):
    def __init__(self):
        super().__init__()

        #self.l1loss = nn.L1Loss()  # *smooth* L1 loss in the paper; see Remarks section in the tutorial
        self.clsloss = nn.BCEWithLogitsLoss()

    def forward(self, pred_cls, pred_inst, inst):

        true_classes = torch.ones((pred_cls.size(0)), dtype=torch.float).to(device)  # (N, 8732)

        true_classes[inst == -1] = 0
 
        inst_loss = instance_loss(pred_inst, inst, margin=0.5)  # (), scalar
        score_loss = self.clsloss(pred_cls, true_classes)
        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)
        print('THE LOSSSSSSSSSSSSSSSSS')
        print(inst_loss)
        print(score_loss)
        return 0.5*score_loss + 0.5*inst_loss
    



################################ MAIN MODEL #####################################################
class WaveSegmentation(pl.LightningModule):
    def __init__(self, in_channels, out_channels, dim_model, batch_size, lr_scheduler_milestones, lr_gamma, nfeatures, nmeasurements, lr=1e-2, L2_reg=1e-3, k=16, loss = SegLoss()):
        super().__init__()
        self.k = k

        self.batch_size = batch_size
        self.nfeatures = nfeatures
        self.nmeasurements = nmeasurements

        
        self.lr = lr
        self.lr_scheduler_milestones = lr_scheduler_milestones
        self.lr_gamma = lr_gamma
        self.L2_reg = L2_reg
        self.loss = loss

        self.train_acc = F1MetricAccumulator()
        self.val_acc = F1MetricAccumulator()

        # Log hyperparams (all arguments are logged by default)
        self.save_hyperparameters(
            'nfeatures',
            'L2_reg',
            'lr',
            'lr_gamma',
            'lr_scheduler_milestones',
            'batch_size'
        )

        self.validation_step_outputs = []
        self.training_step_outputs = []

        # first block
        self.mlp_input = MLP([nmeasurements, dim_model[0]], plain_last=False)

        self.transformer_input = TransformerBlock(
            in_channels=dim_model[0],
            out_channels=dim_model[0],
        )

        # backbone layers
        self.transformers_up = torch.nn.ModuleList()
        self.transformers_down = torch.nn.ModuleList()
        self.transition_up = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        for i in range(0, len(dim_model) - 1):

            # Add Transition Down block followed by a Point Transformer block
            self.transition_down.append(
                TransitionDown(in_channels=dim_model[i],
                               out_channels=dim_model[i + 1], k=self.k))

            self.transformers_down.append(
                TransformerBlock(in_channels=dim_model[i + 1],
                                 out_channels=dim_model[i + 1]))

            # Add Transition Up block followed by Point Transformer block
            self.transition_up.append(
                TransitionUp(in_channels=dim_model[i + 1],
                             out_channels=dim_model[i]))

            self.transformers_up.append(
                TransformerBlock(in_channels=dim_model[i],
                                 out_channels=dim_model[i]))

        # summit layers
        self.mlp_summit = MLP([dim_model[-1], dim_model[-1]], norm=None,
                              plain_last=False)

        self.transformer_summit = TransformerBlock(
            in_channels=dim_model[-1],
            out_channels=dim_model[-1],
        )

        # class score computation
        self.mlp_output_cls = MLP([dim_model[0], 64, out_channels], norm=None)
        self.mlp_output_inst = MLP([dim_model[0], 64, 1], norm=None)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, pos, batch=None):

        # add dummy features in case there is none
        if x is None:
            x = torch.ones((pos.shape[0], 1)).to(pos.get_device())

        out_x = []
        out_pos = []
        out_batch = []

        # first block
        x = self.mlp_input(x)
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

        # summit
        x = self.mlp_summit(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_summit(x, pos, edge_index)

        # backbone up : augment cardinality and reduce dimensionnality
        n = len(self.transformers_down)
        for i in range(n):
            x = self.transition_up[-i - 1](x=out_x[-i - 2], x_sub=x,
                                           pos=out_pos[-i - 2],
                                           pos_sub=out_pos[-i - 1],
                                           batch_sub=out_batch[-i - 1],
                                           batch=out_batch[-i - 2])

            edge_index = knn_graph(out_pos[-i - 2], k=self.k,
                                   batch=out_batch[-i - 2])
            x = self.transformers_up[-i - 1](x, out_pos[-i - 2], edge_index)

        # Class score
        out_cls = self.mlp_output_cls(x)
        out_inst = self.mlp_output_inst(x)


        return self.sigmoid(out_cls), out_inst
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.L2_reg)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_scheduler_milestones, gamma=self.lr_gamma),
            'name': 'LearningRate'
        }
        return [optimizer], [lr_scheduler]
    
    def on_train_start(self):
        # Add mean accuracies per epoch to the hyperparam Tensorboard's tab
        self.logger.log_hyperparams({'hp/train_acc': 0, 'hp/val_acc': 0})



    def training_step(self, batch, batch_idx):
        x, pos, y = batch['x'], batch['pos'], batch['y']
        x = x.view(-1, self.nmeasurements)
        pos = pos.view(-1,3)
        y = y.view(-1)
        predicted_cls, predicted_inst = self(x, pos) 
        predicted_cls = predicted_cls.view(-1)
        predicted_inst = predicted_inst.view(-1)
        # Prior boxes
        train_loss = self.loss(predicted_cls, predicted_inst, y)
##### CHANGE PREDICTION TO THE BOXES AND LABELS THAT HAVE BEEN PREDICTED
        train_acc = self.train_acc.update(predicted_cls, predicted_inst, y)
        
        # Add logs of the batch to tensorboard
        self.log('train_loss', train_loss, on_step=True)
        self.log('train_acc', train_acc, on_step=True)
        
        out = {'loss': train_loss, 'prediction_instances': predicted_inst, 'predicted_scores': predicted_cls, 'target_instances': y}
        self.training_step_outputs.append(out)
        return out


    # This hook receive the outputs of all training steps as a list of dictionaries
    def on_train_epoch_end(self):
        mean_loss = torch.stack([k['loss'] for k in self.training_step_outputs]).mean()
        train_acc = self.train_acc.compute()
        print('THE TRAINING ACCURACYYYYYYYYYY')
        print(train_acc)
        
        self.log('MeanEpoch/train_loss', mean_loss)
        # The .compute() of Torchmetrics objects compute the average of the epoch and reset for next one
        self.log('MeanEpoch/train_acc', train_acc)
        self.log('hp/train_acc', train_acc)

    def validation_step(self, batch, batch_idx):
        x, pos, y = batch['x'], batch['pos'], batch['y']
        x = x.view(-1, self.nmeasurements)
        pos = pos.view(-1,3)
        y = y.view(-1)
        predicted_cls, predicted_inst = self(x, pos) 
        predicted_cls = predicted_cls.view(-1)
        predicted_inst = predicted_inst.view(-1)
        
        # Prior boxes
        val_loss = self.loss(predicted_cls, predicted_inst, y)
##### CHANGE PREDICTION TO THE BOXES AND LABELS THAT HAVE BEEN PREDICTED
        val_acc = self.val_acc.update(predicted_cls, predicted_inst, y)

        # Add logs of the batch to tensorboard
        self.log('MeanEpoch/val_acc', val_acc, on_epoch=True, prog_bar=True)
        self.log('hp/val_acc', val_acc)

        out = {'loss': val_loss, 'prediction_instances': predicted_inst, 'predicted_scores': predicted_cls, 'target_instances': y}
        self.validation_step_outputs.append(out)
        return out

    # This hook receive the outputs of all validation steps as a list of dictionaries
    def on_validation_epoch_end(self):
        mean_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        val_acc = self.val_acc.compute()
        print('THE VALIDATION ACCCCCCCCCCCCCCC')
        print(val_acc)

        self.log('MeanEpoch/val_loss', mean_loss)
        self.log('MeanEpoch/val_acc', val_acc)
        self.log('hp/val_acc', val_acc)

        #self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, pos, y = batch['x'], batch['pos'], batch['y']
        x = x.view(-1, self.nmeasurements)
        pos = pos.view(-1,3)
        y = y.view(-1)
        predicted_cls, predicted_inst = self(x, pos) 
        predicted_cls = predicted_cls.view(-1)
        predicted_inst = predicted_inst.view(-1) 
        return {'predicted_classes': predicted_cls, 'predicted_instances': predicted_inst}



