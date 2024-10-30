# Point/Graph Neural Network for Collective Event Detection in Cell Signalling

We train a neural network to identify waves of intercellular ERK signalling in 2D epithelial monolayers. 
Our previous approach, <a href=https://arcos.gitbook.io>ARCOS</a>, relies on identifying periods of kinase activity in single-cell activity time series obtained from segmentation of time-lapse microscopy images.
Activity identification involves detrending, peak detection and binarisation of the time series, which comes with parameters that require fine-tuning on a case-by case basis. 
Here, we take a different, neural network-based approach and apply it to the segmentation data represented as a point cloud.
Each cell in a field of view (FOV) is described as a point that corresponds to the X/Y location of a centroid of the cell's nucleus. 
The kinase activity is represented as a point's property, i.e., colour, and the evolution of this activity over time occurs along the z-axis of the point-cloud.
The z-axis represents time.

<br>
<p float="left">
    <img src="images/erkktr_full.png" style="width:45%; height:auto;"/>
    <img src="images/erkktr_clouds.png" style="width:45%; height:auto;" />

</p>
<br>
    
## Data Creation

For the purpose of this project we used simulated data created with <a href=https://ccl.northwestern.edu/netlogo/>NetLogo</a>, a multi-agent programmable modeling environment. 
The code to perform the simulations can be found in the `data` folder. 
The shell script `run_netlogo_sim.sh` starts the simulation according to the model specified in `waves-smallFOV.nlogo` and configuration in `wavesim_setup.xml` files.
The script automatically reruns the NetLogo model several times, then adds noise into it and randomly cuts the data such that the point clouds have different sizes. 
Since NetLogo outputs a .csv file for every time point in a model, the shell script executes a Python script that combines all output files into a single file `whole_dataset.csv`. 

A test dataset can be found om `all_data.zip`. 
The Graph Neural Network framework (GNN) only takes a zipped .csv file as input.


## Running the GNN

Once the data has been created, the main model can be run from the `/source_inst/train_model.py` file. 
The adjustable parameters can be found in the beginning of the training file or with `python train_model.py -h`.
To run a basic version of the GNN run `python train_model.py -d ./data/test_data.zip` in terminal after activating the conda environment provided in `pointcnn-env-linux.yml`. 
