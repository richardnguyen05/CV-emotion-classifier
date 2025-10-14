# temp readme.md
    code here

start with placing ravdess wav/mp4 files into raw directory  
- 4M/2F total dataset
- split: 3M/1F train. 1M/1F eval


VISUAL (CNN, use OpenCV to extract frames, and then use pre-trained 2D CNN model. pytorch or keras)

FUSION CLASSIFIER (finish video classifier)

scikit for evaluation, cm f1 and other performance metrics. compare evaluation using just audio or visual only, then compare with multimodal model

### PROCESS
small CNN from scratch 

train on fer 2013 

use 2-3 conv layers, 1-2 FC layers 

data augmentation - helps generalization 
