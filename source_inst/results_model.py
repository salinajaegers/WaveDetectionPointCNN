#### this is done with model.eval() and then save the boxes and visualize them
#### maybe include some stats about the waves? how long they are? following the original one? width?
from utils import *
from data_classes import *
from tqdm import tqdm
from pprint import PrettyPrinter

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader
from models2 import WaveSegmentation
import pytorch_lightning as pl

#from load_data import DataProcesser
#import pandas as pd

import matplotlib 
matplotlib.use('Agg')


# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()



def detect(data, model, meas_names, top_k=20, max_overlap=0.8, min_score=0.5, batch_size=1):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    :param device: device
    """

    # Set model to evaluation mode
    model.eval()
    model.double()
    model.batch_size = 1
    model = model.to(device)
    model = model.float()


    
    test_loader = DataLoader(dataset=data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4)


  
    trainer = pl.Trainer()
    predictions = trainer.predict(model, test_loader)
    

    # Move detections to the CPU
    #det_boxes = det_boxes[0].to('cpu')
    with PdfPages('results.pdf') as pdf:
        for c in range(len(predictions)):
            scores = torch.clone(predictions[c]['predicted_classes'])
            instances = torch.clone(predictions[c]['predicted_instances'])
            instances[scores <= 0.5] = 0
#### TODO: check the clouds input
            x = data[c]['x']
            pos = data[c]['pos']
            # Extracting data for plotting
            x_positions = pos[:, 0].numpy()
            y_positions = pos[:, 1].numpy()
            time_points = pos[:, 2].numpy()
            measurements = x.numpy()

            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # full plot with all the boxes
            scat = ax.scatter(x_positions, time_points, y_positions, c=instances, cmap='hsv', s=20, alpha=0.7)
            plt.colorbar(scat, label='instances')
            ax.set_title('All Boxes. Node Positions and Time Points Colored by Instance')
            ax.set_xlabel('X Position')
            ax.set_zlabel('Y Position')
            ax.set_ylabel('Time Points')
            plt.tight_layout()
            pdf.savefig() 
            plt.close()

            fig = plt.figure()
            ax2 = fig.add_subplot(111, projection='3d')
            # full plot with all the boxes
            scat = ax.scatter(x_positions, time_points, y_positions, c=scores, cmap='hsv', s=20, alpha=0.7)
            plt.colorbar(scat, label='scores')
            ax2.set_title('All Boxes. Node Positions and Time Points Colored by Score')
            ax2.set_xlabel('X Position')
            ax2.set_zlabel('Y Position')
            ax2.set_ylabel('Time Points')

            plt.tight_layout()
            pdf.savefig() 
            plt.close()

        
        # We can also set the file's metadata via the PdfPages object:
        d = pdf.infodict()
        d['Title'] = 'Identified Instances'

    return None





if __name__ == '__main__':
    data_file = './datatest_data.zip'
    model_file = './source_inst/logs/test/2024-09-14-08__04__03_test_data.pytorch'
    meas_var = ['erkactivityvalue']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_top_worst = 5
    batch_size = 800

    model = torch.load(model_file)
    

    # Dataloader
    data = DataProcesserSeg(data_file)
    data.split_data()
    data_test = CloudDatasetSeg(dataset=pd.DataFrame(data.test_set))

    detect(data_test, model, meas_var, top_k=5, max_overlap=0.8, min_score=0.5, batch_size=1)