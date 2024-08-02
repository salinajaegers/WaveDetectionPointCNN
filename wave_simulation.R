library(tidyverse)


path_to_netlogo_output <- '/Users/salina/Downloads/Netlogo3/'


# concatinate all the obersvations into one dataframe
df3 <- list.files(path = path_to_netlogo_output,
           pattern="*.csv", 
           full.names = T) %>% 
      map_df(~read_csv(., col_names = FALSE))

names(df3) <- c('time_point', 'ID', 'xcoordinate', 'ycoordinate', 'apoptosis', 
                     'ERKactivity', 'Survival', 'ERKactivityValue', 'Wave_id', 'Cloud_id')



# eliminate the cells from the point on when they die

# add noise to the positions and the activity values
noise <- as.data.frame(matrix(runif(n=nrow(df3)*3, min=-0.5, max=0.5), nrow=nrow(df3)))

df3$xcoordinate <- df3$xcoordinate + noise$V1
df3$ycoordinate <- df3$ycoordinate + noise$V2
df3$ERKactivityValue <- df3$ERKactivityValue + noise$V3
df3$ERKactivityValue[df3$ERKactivityValue<0] <- 0

df3$Cloud_id <- 3
#df3$Cloud <- rep(1, nrow(df3))




##### FIND THE boxes3 
# format: x_min, y_min, time_min, x_max, y_max, time_max

boxes3 <- data.frame(matrix(ncol = 8, nrow = 0))
colnames(boxes3) <- c('Cloud_id', 'Wave_id', 'x_min', 'y_min', 'time_min', 'x_max', 'y_max', 'time_max')

# remove the background
waves <- unique(df3$Wave_id)
waves <- waves[waves != -1]


# create boxes3 for the waves by ID
for (w in waves){
  wave <- df3[df3$Wave_id == w,]
  cloud <- 3
  
  x_min <- min(wave$xcoordinate)
  y_min <- min(wave$ycoordinate)
  time_min <- min(wave$time_point)
  
  x_max <- max(wave$xcoordinate)
  y_max <- max(wave$ycoordinate)
  time_max <- max(wave$time_point)
  
  if (time_max - time_min > 3){
    wave_box <- data.frame(matrix(c(cloud, w, x_min, y_min, time_min, x_max, y_max, time_max),nrow=1, ncol=8))
    colnames(wave_box) <- c('Cloud_id', 'Wave_id', 'x_min', 'y_min', 'time_min', 'x_max', 'y_max', 'time_max')
    
    boxes3 <- rbind(boxes3, wave_box)
  }
}


write_csv(boxes3, paste0(path_to_netlogo_output, 'boxes3.csv'))





df_full <- rbind(df, df2)
df_full <- rbind(df_full, df3)
write_csv(df_full, paste0('/Users/salina/Downloads/', 'netlogo_simulations.csv'))

boxes_full <- rbind(boxes, boxes2)
boxes_full <- rbind(boxes_full, boxes3)
write_csv(boxes_full, paste0('/Users/salina/Downloads/', 'netlogo_simulations_boxes.csv'))





