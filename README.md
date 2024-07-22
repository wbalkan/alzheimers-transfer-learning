# Transfer Learning with Popular Image Classification Models to Diagnose Alzheimer's Disease
Paper download available [here](https://github.com/user-attachments/files/16338857/COSC78.Final.Paper.docx)

### Description
This repository contains the code for a project which investigates the effectiveness of transfer learning with well-known CNN architectures to classify Alzheimer's disease from MRI scans using the OASIS Alzheimer’s Detection dataset. The architectures explored include ResNet, DenseNet, SqueezeNet, MobileNet, and Vision Transformer.

### Abstract

We explored the efficacy of transfer learning with popular CNN architectures for classifying Alzheimer's from MRI images in the OASIS Alzheimer’s Detection dataset. We tested the following models: Resnet, Densenet, Squeezenet, Mobilenet, and Vision Transformer. We only used 40% of the entire OASIS dataset when creating training/test/validation splits due to compute and time limitations. During experimentation, slight accuracy improvements were observed from data augmentation techniques like cropping, resizing, and hippocampus segmentation. No improvements were seen after oversampling and data trimming/balancing. Overall, the models were largely biased toward predicting Non Demented, even when using oversampling and data trimming/balancing. Vision Transformer performed the best after fine-tuning, achieving an accuracy of 80.80% in Alzhiemer’s multi-class classification. 
