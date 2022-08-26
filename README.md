<img src="http://www.crns.rnrt.tn/front/img/logo.svg">

# Welcome to the official implementation of "Self-Supervised Endoscopic Image Key-Points Matching"
[![arXiv](https://img.shields.io/badge/arXiv-2208.11424-b31b1b.svg)](https://arxiv.org/abs/2208.11424) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<pre>
@article{ref,
    author = {Manel Farhat and Houda Chaabouni{-}Chouayakh and Achraf Ben{-}Hamadou}, 
    title = {Self-Supervised Endoscopic Image Key-Points Matching}, 
    journal = {Elsevier Expert Syst. Appl.},
    year = {2022 Inpress}
}
</pre>
 ![demo_vid](assets/matching_demo.gif)
# Pre-trained models 
Pre-trained models can be downloaded following the link. You may save the models to ./models folder
- https://drive.google.com/drive/folders/1T_DxBmLh7JP-EcUOndiToSdZr6kdoB7m
# Training and testing datasets for endoscopic images matching
- https://drive.google.com/drive/folders/1b9oayHcVPVhLcVOMQTWbIKQVlRv7CBgh

unzip the two zip files in the ./data folder and then update your configuration yaml files accordingly (see blow)
  
# Runners and configurations files
- "**run_trainning.py**", configuration yaml file in "**config/config_train.yaml**"
- "**run_validation.py**", configuration yaml file in "**config/config_validation.yaml**"
- "**run_generate_triplet_dataset.py**", configuration yaml file in "**config/config_triplet_generation.yaml**"
- "**run_matching_demo.py**", configuration yaml file in "**config/matching_demo.yaml**"



## Setup for Dev on local machine
This code base is tested only on Ubuntu 20.04 LTS, TitanV and RTX2080-ti NVIDIA GPUs.
- Install local environment and requirements
First install Anaconda3 then install the requirements as follows:

> **conda create -n crns---self-sup-image-matching python=3.8**

- a new virtual environment is now created in **~/anaconda3/envs/crns---self-sup-image-matching**
Now activate the virtual environment by running:

> **source activate crns---self-sup-image-matching**

- In case you would like stop your venv **`conda deactivate`**

- To install dependencies, cd to the directory where requirements.txt is located and run the following command in your shell:

> **cat requirements.txt  | xargs -n 1 -L 1 pip3 install**

> Install **`docker`** and **`nvidia-docker`** and then run these commands:
>
> **`nvidia-docker build -t <YOUR_DOCKER_IMAGE_NAME>:<YOUR_IMAGE_VERSION> . `**
>
> **`sudo NV_GPU='0' nvidia-docker run  -i -t --entrypoint=/bin/bash --runtime=nvidia <YOUR_DOCKER_IMAGE_NAME>:<YOUR_IMAGE_VERSION>`**
>
- How to use Facial Process on local machine:
> To test photos from local folder: run this command **`python local_processing.py`**
> To process photos coming from the front on your machine: run this command **`python main.py`**


## Git pre-commit hooks
> if not already installed from the requirements.txt then first install pre-commit and black using these commands: **`pip3 install pre-commit`**
> and **`pip3 install black`**

> run **`pre-commit install`** to set up the git hook scripts
>
> You can also **`flake8 <YOURSCRIPT>.py`** to check if your python script is compliant with the project
>
> or directly fix your script using **`black <YOURSCRIPT>.py`**


## Docker build
- [comming soon]

## Known issues
