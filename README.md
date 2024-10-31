<p align="center">
  <a href="https://github.com/RyanQ96/VADIS">
    <img src="./assets/vadis-logo.png" alt="" width="60%" align="top" style="border-radius: 10px; padding-left: 120px; padding-right: 120px; background-color: white;">
  </a>
</p>

This repository contains the code and resources for the paper **"VADIS: A Visual Analytics Pipeline for Dynamic Document Representation and Information-Seeking"**. You can access the paper [here](https://www.computer.org/csdl/journal/tg/5555/01/10677360/209oqMvDHtm).

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Prompt-based Attention Model (PAM)](#training-the-prompt-based-attention-model-pam)
  - [Relevance Preserving Map Projection](#relevance-preserving-map-projection)
- [Contributing](#contributing)
- [License](#license)
- [TODO](#todo)
- [BibTex](#BibTex)
## Features

- **Dynamic Document Embeddings**: Adjust embeddings based on user queries using the Prompt-based Attention Model (PAM).
- **Relevance Visualization**: Visualize documents in a way that reflects both relevance and similarity.
- **Interpretability**: Understand model focus through attention visualization.
- **Extensibility**: Support for multiple datasets and customizable training parameters.

## Project Structure
```md
.
├── arguments.py
├── data/
├── dataloader/
│   ├── dataset.py
├── model/
│   ├── pam.py
│   └── utils.py
├── notebooks/
│   ├── projection_example.ipynb
│   └── setup.py
├── relevance_preserving_map/
│   ├── circular_som.py
├── requirements.txt
├── run.sh
└── train.py
```
- `arguments.py`: Defines training arguments using <code>HfArgumentParser</code>.
- `data/`: Directory for placing additional training data.
- `dataloader/`: Implements data loading for different datasets.
- `model/`: Contains the implementation of the Prompt-based Attention Model (PAM).
- `notebooks/`: Jupyter notebooks for projection examples.
- `relevance_preserving_map/`: Implementation of the Relevance Preserving Map using Circular Self-Organizing Maps.
- `run.sh`: Shell script to run the training with specified parameters.
- `train.py`: Script for training the PAM.

## Usage

### Training the Prompt-based Attention Model (PAM)
The PAM generates dynamic document embeddings and relevance scores based on user queries.

#### Step 1: Prepare Your Data
Place any additional training data in the data/ directory.

#### Step 2: Configure Training Parameters
You can set training parameters via command-line arguments or by editing the run.sh script.
```sh
#!/bin/bash

python train.py \
  --report_name "training_run" \
  --num_epochs 3 \
  --learning_rate 5e-5 \
  --batch_size 4 \
  --max_length 512 \
  --model_name "bert-base-uncased" \
  --datasets "Your dataset"\
  --use_dual_loss True \
  --entropy_weight 0.01 \
  --load_pretrained_model False \
  --pretrained_model_path "" \
  --model_save_path "./models"
```

### Relevance Preserving Map Projection
The Relevance Preserving Map uses a Circular Self-Organizing Map (SOM) to visualize data distribution, balancing relevance and semantic similarity between data. This projection method can be applied broadly to data distribution with relevance information. 

<strong>Key Features</strong>
* User-Driven Relevance Adjustment: Documents are placed based on dynamic relevance scores that adapt to the user’s query.
* Circular Layout: Documents are visualized on a circular grid, balancing relevance and similarity.

Here's how to use the Circular SOM to generate a projection of your documents.

<strong>Sample Code for Running the Projection</strong>
Here's how to use the Circular SOM to generate a projection of your documents.
```py
import numpy as np
from relevance_preserving_map.circular_som import CircularSOM, get_grid_position_som, plot_som_results

# Sample data: Replace with your document embeddings and relevance scores
data = np.random.rand(100, 300)  # Example: 100 documents, 300 features each
relevance = np.random.rand(100)  # Relevance scores for each document
labels = np.arange(0, 100)       # Labels or identifiers for each document

# Initialize Circular SOM
som = CircularSOM(
    step=8,                       # Number of neurons in the first layer
    layer=21,                     # Number of layers in the circular grid
    input_len=data.shape[1],      # Input dimensionality
    sigma=1.5,                    # Initial neighborhood size
    learning_rate=0.7,            # Initial learning rate
    activation_distance='euclidean',
    topology='circular',
    neighborhood_function='gaussian',
    random_seed=10
)

# Train the SOM
som.train(
    data=data,
    relevance_score=relevance,
    num_iteration=1000,  # Adjust as needed
    w_s=0.2,             # Weight for similarity
    w_r=0.8,             # Weight for relevance
    verbose=True,
    report_error=True,
    use_sorted=True
)

# Get grid positions after training
ids_same_order = np.arange(data.shape[0])
data_grid_positions = get_grid_position_som(som, data, relevance, ids_same_order)

# Visualize the results
plot_som_results(som, data, labels, relevance, sort=True)
```
<strong>Notebook</strong>
For a detailed example, refer to the Jupyter Notebook [`notebooks/projection_example.ipynb`](notebooks/projection_example.ipynb)

## TODO 
- [ ] Open-source the component-based frontend system 

## BibTeX
```bibtex
@article{qiu2024vadis,
  title={VADIS: A Visual Analytics Pipeline for Dynamic Document Representation and Information-Seeking},
  author={Qiu, Rui and Tu, Yamei and Yen, Po-Yin and Shen, Han-Wei},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2024},
  publisher={IEEE},
  doi={10.1109/TVCG.2024.10677360},  % You can replace this with the actual DOI if available
  url={https://www.computer.org/csdl/journal/tg/5555/01/10677360/209oqMvDHtm}
}

```
