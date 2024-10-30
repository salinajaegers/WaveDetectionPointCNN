#!/bin/bash


# Path to your NetLogo model
model_path="./data/waves-smallFOV.nlogo"

# Output directory
output_dir="./data/Netlogo_output"
mkdir "${output_dir}"

# Setup File
setup_file="./data/wavesim_setup.xml"

# Loop over the model values -> how many point clouds will be generatated
for model in $(seq 1 20)
do
    mkdir "${output_dir}${model}"
    cd "${output_dir}${model}"
    cp "${model_path}" "${output_dir}${model}"
    "/Volumes/NetLogo 6.4.0 1/NetLogo 6.4.0/netlogo-headless.sh" --headless\
	--model "${output_dir}${model}/waves-smallFOV.nlogo" \
	--table - \
	--setup-file ${setup_file}
	
done


python ./data/wave_prep.py
