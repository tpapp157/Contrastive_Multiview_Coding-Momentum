# Contrastive_Multiview_Coding-Momentum
Tensorflow 2 implementation of:

[Contrastive Multiview Coding](https://arxiv.org/abs/1906.05849)

[Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)

Based off the [official PyTorch implementation](https://github.com/HobbitLong/CMC).


## Notes
Tensorflow doesn't seem to support editing and maintaining variables across training steps. Due to this the momentum memory queue needs to be returned and resubmit with each training step. As a result, the current implementation does not support continuing training seamlessly from a checkpoint as the memory will be reinitialized with random numbers (could bypass this by manually writing/reading to file).

## Example
Using the [Terrain Map dataset](https://www.kaggle.com/tpapp157/earth-terrain-height-and-segmentation-map-images) which I uploaded to Kaggle for demonstration. The dataset provides 512x512 images of terrain, height, and segmentation data. Paired 256x256 crops are taken from each as well as the 512 terrain image resized to 256 are used as four views for CMC.

Each view learns its own encoding network although for simplicity the network structure is the same for each. Network structure consists of an initial convolution, 5 downscaling residual blocks, and a final embedding convolution and global mean pooling. 128 dimensions are used for the final CMC embedding to calculate NCE loss.

Normalized Convolutions are used throughout the network. These are inspired by the StyleGAN2 implementation but generalized to be a dropin replacement for a normal convolution layer. The Normalized Convolution normalizes the convolution kernel so that feature metrics don't compound and explode over many layers. This eliminates the need for Batch Normalization by baking data normalization into the convolution kernel and allowing the model to learn the proper dataset normalization statistics (rather than simply calculating batch statistics each training step). Training collapses without normalization.

Training for 15 Epochs and using UMAP to visualize the CMC embeddings produces this plot:
![](../master/images/Embed.png)

Clearly the CMC embeddding has learned the general data structure. It's likely that outputs from earlier convolutional layers would be more informative depending on the detail scale of interest.


## Update: Unified Feature Extractor
As shown in [Rethinking Few-Shot Image Classification: a Good Embedding Is All You Need?](https://arxiv.org/abs/2003.11539), using a single feature extraction network for all views can improve the overall quality of features learned. Training a single network also allows that network to be significantly larger.

CMC_combined implements a 34-layer resnet as the core feature extractor for all image views. Since each view starts with a different number of channels (3 for terrain, 1 for height, 7 for segmentation) each view learns its own initial convolution layer to standardize the number of channels prior to the feature extractor. Similarly, each view learns its own final fully-connected classification layer.

Learned embeddings are improved with more detailed structure. Training is significantly more stable.
