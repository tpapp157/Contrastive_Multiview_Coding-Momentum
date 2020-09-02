# Contrastive Learning of Doom Eternal

## Goal
This script explores using Contrastive Learning to derive image features sufficient for basic image segmentation in a completely unsupervised pipeline.

## Dataset
Images were extracted from a video of a full playthrough of the game Doom Eternal. The video spanned ~12 hours of gameplay and frames were extracted every 0.1 seconds resulting in a total of ~400k sequential images with a resolution of 1920x1080 pixels.

Despite the large amount of data, this dataset still presents significant challenges due to the high diversity and large biases among image content. Gameplay spans many visually distinct world environments as well as cinematics and menu interfaces. While some objects such as common enemies and menu interfaces appear regularly throughout the dataset, many others are unique and appear only briefly. Combat is very fast paced and visually noisy but makes up only a small fraction of the overall dataset which is otherwise dominated by simpler environment traversal and menu navigation. Unlike many public image datasets which are currated into neat categories and whose images are framed and centered to only feature the object of interest, the Doom images often feature many objects with conflicting intersections and occlusions and cutoffs at image edges.

## Pipeline
I trained a 34 layer ResNet via Self-Supervised Contrastive Learning. Four separate views are extracted for each image: concentric large, medium, and small crops as well as a corresponding large crop from the next image in the sequence. Loss metrics are used to guide image sampling in a self-supervised manner (images with higher loss metrics are sampled proportionally more frequently). Image batches are currated where the first image is sampled randomly and the remainder of the batch is sampled from other nearby images in the dataset sequence. A memory queue and momentum network are also used to increase diversity and stabilize training.

Once the network has been trained, a sequence of images are selected and processed by the network at multiple resolution scales. The resulting feature maps are resized and concatenated. This image feature dataset is then post-processed, dimension reduced with UMAP, and clustered with HDBSCAN to extract the final segmentation. A short sequence is shown below.

## Sample Segmentation Result
![](images/DoomSegmentation.gif)
