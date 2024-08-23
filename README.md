### Point/Graph CNN for Wave Detection in Cell Signalling Dynamics

## Input format:
The point cloud of cells has the dimensions: 
        x - xcoordinate in image
        y - time points
        z - ycoordinate in image

Two files need to be passed to the GNN:
- 'boxes.csv': In this file are the annotated boxes for the training. The format is
  'Cloud_id', 'Wave_id', 'width', 'length', 'height', 'center_x', 'center_y', 'center_z', 'euler_z', 'euler_y', 'euler_x'
- 'dataset.csv': The file with all the point clouds. The format is:
  'Cloud_id', 'Wave_id', 'time_point', 'xcoordinate', 'ycoordinate', 'measurement_1', 'measurement_2', etc.

The boxes can be created from annotated cells in a wave with the wave_simulation.ipynb notebook.
