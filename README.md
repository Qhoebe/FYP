# FYP on Smart Object Counter

Developing a smart object counter capable of detecting and counting objects in images. The solution is split into 2 modules -- a backend server and a frontend web application. 

The backend is implemented using Flask, serving as the core processing unit for object counting. 

The frontend is developed using React. The web application allows users to view a live camera feed and initiate the object-counting process by pressing a button. 

## Features
* Live camera feed display
* Start/Stop buttons to control model evaluation
* flask-based backend server that executes model evaluation

## Installation and Setup
Set up the environment by installing the required libraries

### Set Up Model Training Environment with **NVIDIA GPUs and CUDA support**

**Prerequisite**: Have Anaconda installed  

1. Create the conda environment
```
conda create -n fypEnv python=3.8 -y 
```

2. Activate the environment
```
conda activate fypEnv 
```

3. Install the following libraries
```
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia 
conda install flask flask_cors numpy opencv -c conda-forge
conda install transformers
conda install pycocotools tqdm
```

### Set Up Backend Environment with **NVIDIA GPUs and CUDA support**

**Prerequisite**: Have Anaconda installed  

1. Create the conda environment
```
conda create -n fypBackend python=3.8 -y 
```

2. Activate the environment
```
conda activate fypBackend 
```

3. Install the following libraries
```
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia 
conda install flask flask_cors numpy opencv -c conda-forge
conda install transformers
```

### Start Frontend Project

**Prerequisite**: Download Node.js 

1. Install dependencies
```
npm install
```

2. Start project
```
npm start
```






