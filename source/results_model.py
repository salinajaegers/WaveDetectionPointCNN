#### this is done with model.eval() and then save the boxes and visualize them
#### maybe include some stats about the waves? how long they are? following the original one? width?
from utils import *
from tqdm import tqdm
from pprint import PrettyPrinter

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader

from load_data import DataProcesser
import pandas as pd

import matplotlib 
matplotlib.use('Agg')


# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()



def detect(clouds, model, meas_names, top_k=20, max_overlap=0.8, min_score=0.5, batch_size=10):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    :param device: device
    """


    # Set the CPU/GPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Set model to evaluation mode
    model.eval()
    model.double()
    model.batch_size = 1
    model = model.to(device)


    # Dataloader
    test_loader = DataLoader(dataset=clouds,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4)


    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_scores = list()

##### TODO: i hope this is the right output, check in testing
    predicted_locs, predicted_scores = model(test_loader)

    
    # Detect objects in SSD output
    det_boxes, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)


    # Move detections to the CPU
    #det_boxes = det_boxes[0].to('cpu')
    with PdfPages('identified_boxes.pdf') as pdf:
        for c in range(det_boxes.size(0)):
            boxes = det_boxes[c].to('cpu').to_list()
            scores = det_scores[c].to('cpu').to_list()
#### TODO: check the clouds input
            cloud = clouds[c]
            # Extracting data for plotting
            x_positions = cloud.pos[:, 0].numpy()
            y_positions = cloud.pos[:, 1].numpy()
            time_points = cloud.pos[:, 2].numpy()
            measurements = cloud.x[:, :].numpy()


            fig = plt.figure(projection='3d')
            ax = fig.add_subplot(measurements.size(1), projection='3d')
            for j in range(measurements.size(1)):
                # full plot with all the boxes
                scat = ax[j].scatter(x_positions, time_points, y_positions, c=measurements[:,j], cmap='seismic', s=50, alpha=0.7)
                plt.colorbar(scat, label=str(meas_names[j]))
                ax[j].set_title('All Boxes. Node Positions and Time Points Colored by ' + str(meas_names[j]))
                ax[j].set_xlabel('X Position')
                ax[j].set_zlabel('Y Position')
                ax[j].set_ylabel('Time Points')

                for box in boxes:
                    box_fig(box, ax[j])

            plt.tight_layout()
            pdf.savefig() 
            plt.close()


            
                
            for i in range(boxes.size(0)):
                fig = plt.figure()
                ax = fig.add_subplot(measurements.size(1), projection='3d')
                plt.tight_layout()
                for j in range(measurements.size(1)):

                    box = rotation_to_vertex(boxes[i])
                    xmin, xmax = torch.min(box[0,:]), torch.max(box[0,:])
                    ymin, ymax = torch.min(box[1,:]), torch.max(box[1,:])
                    zmin, zmax = torch.min(box[2,:]), torch.max(box[2,:])

                    # Scatter plot of node positions with time as the third axis, colored by ERKKTR_ratio
                    scat = ax[j].scatter(x_positions[x_positions <= xmax and x_positions >= xmin], 
                                           time_points[time_points <= ymax and time_points >= ymin], 
                                           y_positions[y_positions <= zmax and y_positions >= zmin], 
                                           c=measurements[:,j], cmap='seismic', s=50, alpha=0.7)
                    plt.colorbar(scat, label=str(meas_names[j]))

                    ax[j].set_title('Box ' + str(i) + 'Probability: ' + str(scores[i]) + '. Node Positions and Time Points Colored by ' + str(meas_names[j]))
                    ax[j].set_xlabel('X Position')
                    ax[j].set_zlabel('Y Position')
                    ax[j].set_ylabel('Time Points')


                pdf.savefig() 
                plt.close()
        
        # We can also set the file's metadata via the PdfPages object:
        d = pdf.infodict()
        d['Title'] = 'Identified boxes'


    
    
        
        
        
    


    return None


def box_fig(box, ax):
    if box.size(0) == 1:
        box = rotation_to_vertex(box)

    assert box.size(0) == 8
    assert box.size(1) == 3
    
    # z1 plane boundary
    ax.plot(box[0, 0:2], box[1, 0:2], box[2, 0:2], color='black')
    ax.plot(box[0, 1:3], box[1, 1:3], box[2, 1:3], color='black')
    ax.plot(box[0, 2:4], box[1, 2:4], box[2, 2:4], color='black')
    ax.plot(box[0, [3,0]], box[1, [3,0]], box[2, [3,0]], color='black')

    # z2 plane boundary
    ax.plot(box[0, 4:6], box[1, 4:6], box[2, 4:6], color='black')
    ax.plot(box[0, 5:7], box[1, 5:7], box[2, 5:7], color='black')
    ax.plot(box[0, 6:], box[1, 6:], box[2, 6:], color='black')
    ax.plot(box[0, [7, 4]], box[1, [7, 4]], box[2, [7, 4]], color='black')

    # z1 and z2 connecting boundaries
    ax.plot(box[0, [0, 4]], box[1, [0, 4]], box[2, [0, 4]], color='black')
    ax.plot(box[0, [1, 5]], box[1, [1, 5]], box[2, [1, 5]], color='black')
    ax.plot(box[0, [2, 6]], box[1, [2, 6]], box[2, [2, 6]], color='black')
    ax.plot(box[0, [3, 7]], box[1, [3, 7]], box[2, [3, 7]], color='black')

    return None