# Vision-Transformer (WIP)

This repo contains my Vision-Transformer implementation (WIP).
We are going to implement the basic ViT first. Then we will try to train it on the Imagenette dataset. After that we will try to add some tricks to improve the performance on small dataset (because Imagenette is pretty small).

# References 

* [Paper](https://arxiv.org/pdf/2010.11929.pdf)


# TODOs 

* finish refactoring and config stuff 
* add bash script to download imagenette 
* add image augementation (albumentations)
* run multiple experiments with different hyperparameter combinations 
* add multi-gpu support 
* write down some stuff about the model here 
* add evaluation plots 
* provide a download-link to a trained model 
* instead of using the Pytorch Transformer Encoder blocks, implement them from scratch (for learning)
* research and try some tricks to improve performance on small datasets like this 