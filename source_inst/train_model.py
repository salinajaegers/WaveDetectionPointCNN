# import libraries
import argparse
import datetime
import os
import sys
import time
import warnings
import zipfile

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib 
matplotlib.use('Agg')

import pytorch_lightning as pl

#import torch.optim
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from torch.utils.data import DataLoader


# import custom libraries
path_to_module = './source2'  # Path where all the .py files are, relative to the notebook folder
sys.path.append(path_to_module)
from models2 import WaveSegmentation
from data_classes import DataProcesserSeg, CloudDatasetSeg
from utils import *
from results_model import *



def make_parser():
    parser = argparse.ArgumentParser(description='Train a ConvNetCam model with Adam optimizer and MultiStep learning rate reduction.')

    # For the model
    #parser.add_argument('-c', '--nclouds', help='Number of clouds in the data. If None, automatically detects the number of unique values in the cloud column of the dataset. Default: None.', type=int, default=None)
    parser.add_argument('-f', '--nfeatures', help='Number of features of the input representation before classification. Hint: [5-10], increase if model underfits, decrease if model overfits. Default: 10.', type=int, default=10)
    parser.add_argument('-b', '--batch', help='Batch size. Hint: usually a power of 2. Default: 12.', type=int, default=3)
    parser.add_argument('-r', '--lr', help='Initial learning rate. Default: 0.01', type=float, default=0.1) # originally 0.01
    parser.add_argument('-s', '--schedule', help='Scheduled epochs at which the learning rate will be reduced. If None, will evenly divide the number of epochs into 3. Default: None.', type=int, nargs='+', default=None)
    parser.add_argument('-g', '--gamma', help='Factor by which the learning rate is multiplied at the specified epochs. Default: 0.1.', type=float, default=0.01)
    parser.add_argument('-p', '--penalty', help='L2 penalty weigth. Hint: increase if model overfits. Default: 0.001.', type=float, default=0.001)
    # For the data
    parser.add_argument('-d', '--data', help='Path to the data zip file.', type=str, default=None)
    parser.add_argument('-m', '--measurement', help='Names of the measurement variables. In DataProcesser convention, this is the prefix in a column name that contains a measurement (time being the suffix). Pay attention to the order since this is how the dimensions of a sample of data will be ordered (i.e. 1st in the list will form 1st row of measurements in the sample, 2nd is the 2nd, etc...) If None, DataProcesser will extract automatically the measurement names and use the order of appearance in the column names. Default: None.', type=str, nargs='+')
    # For the trainer
    parser.add_argument('-e', '--nepochs', help='Number of training epochs. Default: 10.', type=int, default=3)
    parser.add_argument('--devices', help='Number of devices to use for training. Default: 7.', type=int, default=10)
    parser.add_argument('--accelerator', help='To use GPU or CPU. Default: CPU', type=str, default='cpu')
    
    # Logs and reproducibility
    parser.add_argument('--logdir', help='Path to directory where to store the logs and the models. Default: "./logs/"', type=str, default='./logs/')
    parser.add_argument('--seed', help='Seed random numbers for reproducibility. Default: 7.', type=int, default=7)

#### TODO: PUT FTRAIN, ECT AND KNNRADIUS IN THE PARSER TOO, and also normalize_meas(method=args.method, norm_per_meas=args.norm_per_meas)
    # args.method, norm_per_meas=args.norm_per_meas
    # 'center', 'zscore', 'minmaxscaling'
    parser.add_argument('--method', help='Normalization method. Choose from: center, zscore, minmaxscaling', type=str, default='center')
    parser.add_argument('--norm_per_meas', help='Normalization method. True/False', type=bool, default=False)

    # Add the flags of pytorch_lightning trainer
    #parser = pl.Trainer.add_argparse_args(parser)

    return parser


def make_loggers(args, dir_logs='logs/', subdir_logs=None, file_logs=None):
    """
    :param dir_logs: directory path to store all the individual models
    :param subdir_log: directory path to store one individual model, is done based on what time the model was run
    :param file_logs: file name of the model
    :return: two loggers: one in csv format, one in tensorboard format (later the model will be saved in pytorch format)
    """
    file_data = os.path.splitext(os.path.basename(args.data))[0]  
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H__%M__%S')

    dir_logs = dir_logs
    if subdir_logs is None:
        subdir_logs = '_'.join(args.measurement)
    if file_logs is None:
        file_logs =  '_'.join([timestamp, file_data])

    if not os.path.exists(dir_logs):
        os.makedirs(dir_logs)

    csv_logger = pl.loggers.CSVLogger(save_dir=dir_logs, name=subdir_logs, version=file_logs)
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=dir_logs, name=subdir_logs, version=file_logs, default_hp_metric=False)

    return csv_logger, tb_logger




def make_configs(args):
    # Convert args to dictionary
    dargs = vars(args)

    config_model = {
    'nfeatures': dargs['nfeatures'],
    'batch_size': dargs['batch'],
    'lr': dargs['lr'],
    'lr_scheduler_milestones': dargs['schedule'],
    'lr_gamma': dargs['gamma'],
    'L2_reg': dargs['penalty']
    }

    config_data = {
    'data_file': dargs['data'],
    'meas_var': dargs['measurement']
    }

    # Add the flags of pytorch_lightning trainer, so can use any option of pl
    custom_keys = ['nfeatures','batch', 'lr', 'schedule', 'gamma', 'penalty', 
                   'data', 'measurement', 'devices', 'nepochs', 'logdir', 'seed', 
                   'accelerator', 'method', 'norm_per_meas']
    pl_keys = set(dargs.keys()).difference(custom_keys)

    config_trainer = {k:dargs[k] for k in pl_keys}
    # Overwrite the default with manually passed values
    config_trainer['max_epochs'] = dargs['nepochs']
    config_trainer['min_epochs'] = dargs['nepochs']
    config_trainer['devices'] = dargs['devices']
    config_trainer['accelerator'] = dargs['accelerator']

    return config_model, config_data, config_trainer


def load_data(args):
    ### LOADING THE DATA
    # custom load
    data = DataProcesserSeg(args.data, datatable=False)

    # Select measurements and times, subset classes and split the dataset
    meas_groups = data.groups if args.measurement is None else args.measurement
    n_groups = len(meas_groups)
    # Auto detect
    nclouds = data.nclouds
    
    # split into train, test, validate
    data.split_data()




    # load the graph sets
    data_train = CloudDatasetSeg(dataset=pd.DataFrame(data.train_set), method=args.method, norm_per_meas=args.norm_per_meas)
    data_validation = CloudDatasetSeg(dataset=pd.DataFrame(data.validation_set), method=args.method, norm_per_meas=args.norm_per_meas)
    
    #data_train.euler_stats
    #data_train.dimension_stats

    if args.batch > len(data_train) or args.batch > len(data_validation):
        raise ValueError('Batch size ({}) must be smaller than the number of clouds in the training ({}) and the validation ({}) sets.'.format(args.batch, len(data_train), len(data_validation)))


    # torch dataloader setup
    train_loader = DataLoader(
        dataset=data_train,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.devices
        #drop_last=True
    )

    validation_loader = DataLoader(
        dataset=data_validation,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.devices
        #drop_last=True
    )

    # define output dictionary
    out = {
        'train_loader': train_loader,
        'validation_loader': validation_loader,
        'measurement': meas_groups,
        'n_groups': n_groups,
        'nclouds': nclouds
    }

    return out


#### TODO: figure out how tf the boxes should be loaded
def train(config_model, config_trainer, train_loader, validation_loader, file_model_csv, file_model_tensorboard):
    """
    Training. Initiate the model and start looping over epochs
    """
    # Set device (cuda or no cuda)
    model = WaveSegmentation(**config_model)
    #WaveSegmentation(3, out_channels=1, dim_model=[32, 64, 128, 256, 512],k=16).to(device)
    
    # Set costum loss criterion for the boxes 
###### TODO: NEEDS TO BE DONE DIRECTLY IN PL MODEL
    #criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)
    #optimizer
    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    trainer = pl.Trainer(**config_trainer)
    trainer.fit(model, train_loader, validation_loader)
    torch.save(model, file_model_tensorboard)
    torch.save(model, file_model_csv)

def main():
    parser = make_parser()
    args = parser.parse_args()
    pl.seed_everything(args.seed, workers=True)
    config_model, config_data, config_trainer = make_configs(args)
    # load the data and its stats
    loaders = load_data(args)

    train_loader = loaders['train_loader']
    validation_loader = loaders['validation_loader']
    measurement = loaders['measurement']
    n_groups = loaders['n_groups']
    nclouds = loaders['nclouds']
    # initialize the loggers
    csv_logger, tb_logger = make_loggers(
        args,
        dir_logs=args.logdir,
        subdir_logs='test'
    )
    # Save the final model in a pytorch format
    file_model_csv = csv_logger.log_dir + '.csv'
    file_model_tensorboard = tb_logger.log_dir + '.pytorch'

    # Update the defaults
    update_model = {}
    update_model['nmeasurements'] = n_groups
    update_model['in_channels'] = 3
    update_model['out_channels'] = 1 # number of classes
    update_model['dim_model'] = [32, 64, 128, 256, 512]
    #in_channels, out_channels, dim_model

    
    if args.schedule is None:
        update_model['lr_scheduler_milestones'] = even_intervals(args.nepochs, ninterval=3)
    config_model.update(update_model)

    update_trainer = {
        'callbacks': [LearningRateMonitor(logging_interval='epoch')],
        'log_every_n_steps': 1,
        'logger': [csv_logger, tb_logger],
        'benchmark': True
    }
    config_trainer.update(update_trainer)

    t0 = time.time()
    train(config_model, config_trainer, train_loader, validation_loader, file_model_csv=file_model_csv, file_model_tensorboard=file_model_tensorboard)
    t1 = time.time()
    print('Elapsed time: {:.2f} min'.format((t1 - t0)/60))
    print('Model saved at: {}'.format(file_model_tensorboard))



    #detect(clouds, model, meas_names, top_k=20, max_overlap=0.8, min_score=0.5, batch_size=10)


if __name__ == '__main__':
    main()