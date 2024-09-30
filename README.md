# ATML-Biases
Assignment for ATML. Discovers the biases of various image classification models. Below is a description of each script file (run the script file which matches you tasks the best):
1. **train.py:** Finetunes the classifier head of resnet-18 or resnet-34 or resnet-50 and reports validation stats on cifar-10, cifar-100, PACS or SVHN.
2. **custom_infer.py:** Finetunes the classifier head of resnet-18 or resnet-34 or resnet-50 using a custom training set and reports validation stats on a custom validation set. YOU MUST GIVE A PATH TO THESE CUSTOM DATASETS TO USE THIS SCRIPT.
3. **noise_inject.py:** Finetunes classifier head of resnet-18 or resnet-34 or resnet-50 on MNIST/CIFAR-10/CIFAR-100. Runs validation on normal and noise injected validation sets and displays the results.
4. **clip_infer.py** Evaluates CLIP on MNIST/CIFAR-10/CIFAR-100/PACS/SVHN and all the other augmented datasets used for Shape/Texture/Color Biases.


## **Datasets**
The attached link below has datasets for the following
1. Stylized Imagenette
2. Stylized CIFAR10
3. Edge Imagenette
4. Colored Imagenette
Drive Link: https://drive.google.com/drive/folders/1Le2XAn74TEtBIJe1jdTojg6gZR5eumHL
