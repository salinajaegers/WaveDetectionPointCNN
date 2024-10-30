library(tidyverse)
library(scatterplot3d)

# function to load netlogo data by trial
netlogo_data <- function(path_to_netlogo_output, cloud_id){
  # concatinate all the obersvations into one dataframe
  df <- list.files(path = path_to_netlogo_output,
                    pattern="*.csv", 
                    full.names = T) %>% 
    map_df(~read_csv(., col_names = FALSE))
  
  names(df) <- c('time_point', 'ID', 'xcoordinate', 'ycoordinate', 'apoptosis', 
                  'ERKactivity', 'Survival', 'ERKactivityValue', 'Wave_id', 'Cloud_id')
  
  
  # add noise to the positions and the activity values
  noise <- as.data.frame(matrix(runif(n=nrow(df)*3, min=-0.5, max=0.5), nrow=nrow(df)))
  
  df$xcoordinate <- df$xcoordinate + noise$V1
  df$ycoordinate <- df$ycoordinate + noise$V2
  df$ERKactivityValue <- df$ERKactivityValue + noise$V3
  df$ERKactivityValue[df$ERKactivityValue<0] <- 0
  
  df$Cloud_id <- cloud_id
  #df$Cloud <- rep(1, nrow(df))
  
  
  
  
  ##### FIND THE boxes 
  # format: x_min, y_min, time_min, x_max, y_max, time_max
  
  boxes <- data.frame(matrix(ncol = 8, nrow = 0))
  colnames(boxes) <- c('Cloud_id', 'Wave_id', 'x_min', 'y_min', 'time_min', 'x_max', 'y_max', 'time_max')
  
  # remove the background
  waves <- unique(df$Wave_id)
  waves <- waves[waves != -1]
  
  
  # create boxes for the waves by ID
  for (w in waves){
    wave <- df[df$Wave_id == w,]
    
    x_min <- min(wave$xcoordinate)
    y_min <- min(wave$ycoordinate)
    time_min <- min(wave$time_point)
    
    x_max <- max(wave$xcoordinate)
    y_max <- max(wave$ycoordinate)
    time_max <- max(wave$time_point)
    
    if (time_max - time_min > 3){
      wave_box <- data.frame(matrix(c(cloud_id, w, x_min, y_min, time_min, x_max, y_max, time_max),nrow=1, ncol=8))
      colnames(wave_box) <- c('Cloud_id', 'Wave_id', 'x_min', 'y_min', 'time_min', 'x_max', 'y_max', 'time_max')
      
      boxes <- rbind(boxes, wave_box)
    }
  }
  
  write_csv(boxes, paste0(path_to_netlogo_output, 'minmaxboxes.csv'))
  
  
  return(list(df, boxes))
}



#df_full <- rbind(df, df2)
#df_full <- rbind(df_full, df)
#write_csv(df_full, paste0('/Users/salina/Downloads/', 'netlogo_simulations.csv'))
#
#boxes_full <- rbind(boxes, boxes2)
#boxes_full <- rbind(boxes_full, boxes)
#write_csv(boxes_full, paste0('/Users/salina/Downloads/', 'netlogo_simulations_boxes.csv'))


path_to_netlogo_output <- '/Users/salina/Desktop/PointCNN/simulation_data/Netlogo1'
cloud1 <- netlogo_data(path_to_netlogo_output, 1)
df <- cloud1[[1]]
boxes <- cloud1[[2]]

awave <- df[df$Wave_id == 412, ]

p3 <- scatterplot3d(awave$xcoordinate, awave$time_point, awave$ycoordinate, xlim=c(0,20), zlim=c(0, 20), ylim=c(60, 85))
#p3$points3d(x=c(7.5745065,14.4435660),y=c(5.6522309,12.4991738),z=c(68,77),type="l")#, col='red')
p3$points3d(x=c(7.57,7.57),y=c(5.65,5.65),z=c(68,77),type="l", lw=2)#, col='red')


library(rgl)

open3d()

# create and plot a box at (x,y,z) of size (x1,y1,z1)

mycube <- box3d()                      # create a cube as mesh object   
mycube <- scale3d(mycube, x1, y1, z1)   # now scale that object by x1,y1,z1
mycube <- translate3d(mycube, x, y, z)  # now move it to x,y,z
wire3d(mycube)                          # now plot it to rgl as a wireframe




axes3d()  # add some axes


