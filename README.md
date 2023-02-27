# Multi-Modal News consistency using Self supervised

# learning

A breakdown of the code base will be provided here. All the code is inherited from the original LEWEL
paper and the github repository (https://github.com/LayneH/LEWEL). My contributions were to update the
default ResNet encoder to CLIP image and text encoder and making sure the code runs in CCR. This included
changing the CLIP model to remove unused parameters so that Distributed Data Parallelism would work as
usual. Following is a breakdown of the codebase :

- Backbone: Contains the code for the updated CLIP model as well as ResNet encoder.
- Data: Contains the code for data loader for the newsclippings dataset.
- Models: Contains the code for the updated LEWEL model.
- Scripts : Contains the code for slurm scripts to be run.
- Utils : Contains code for logging, learning rate update function and code to run for validation error.
- The main.py file of the code to run to train the updated LEWEL model
- Train.py used for training the Downstream classifier model.


