# Contrastive Learning of Doom Eternal

## Goal
This script explores using Contrastive Learning to derive image features sufficient for basic image segmentation in a completely unsupervised pipeline.

## Dataset
Images were extracted from a video of a full playthrough of the game Doom Eternal. The video spanned ~12 hours of gameplay and frames were extracted every 0.1 seconds resulting in a total of ~400k sequential images with a resolution of 1920x1080 pixels.

Despite the large amount of data, this dataset still presents significant challenges due to the high diversity and large biases among image content. Gameplay spans many visually distinct world environments as well as cinematics and menu interfaces. While some objects such as common enemies and menu interfaces appear regularly throughout the dataset, many others are unique and appear only briefly. Combat is very fast paced and visually noisy but makes up only a small fraction of the overall dataset which is otherwise dominated by simpler environment traversal and menu navigation. Unlike many public image datasets which are currated into neat categories and whose images are framed and centered to only feature the object of interest, the Doom images often feature many objects with conflicting intersections and occlusions and cutoffs at image edges.

## Sample Result
![](../master/Doom/images/DoomSegmentation.gif)