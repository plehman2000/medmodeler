# medmodeler
![alt text](https://github.com/plehman2000/MRI/blob/main/assets/MM_logo.png?raw=true)

This is the code for **medmodeler**, an end to end automatic segmentation tool for MRI files, and can identify lungs, bladder, liver, kidney and bone.
 
 ### Usage
For use in your own project, simply download the project, install its dependencies and import "inference()" from inference.py file
### Examples
![alt text](https://github.com/plehman2000/MRI/blob/main/assets/example3.png?raw=true)

![alt text](https://github.com/plehman2000/MRI/blob/main/assets/3dSlicer_example.png?raw=true)

### Architecture
![alt text](https://github.com/plehman2000/MRI/blob/main/assets/fpn.png?raw=true)
* This project utilizes the famous PyTorch segmentation repo by qubvel that can be found [here](https://github.com/qubvel/segmentation_models.pytorch)
* Specifically, this model uses the [Feature Pyramid Network Architecture](https://arxiv.org/abs/1612.03144), with the Mix Vision Transformer as the encoder and pre-training performed on ImageNet

### Dataset
* This model was trained on the [CT-ORG Dataset](https://www.nature.com/articles/s41597-020-00715-8)

### Room for Improvement
* The model performs well on bone and liver segmentation, but often fails to fully capture the other classes, most likely due to insufficient training
* When inferencing, the *tensor_to_label_map()* function is a huge bottleneck, and could likely be implemented more efficiently with matrix operations
* No input validation, inference only accepts traditional medical image formats like *.nii.gz

### Credit
This project was produced in collaboration with the University of Florida, who supplied compute resources in the form of 4 NVidia A100 GPUs.
