import torch
import torch.nn as nn
import torchvision.models as models
import clip

def load_resnet_ft(model_name, ft_type):
    if model_name == 'resnet-18':
        model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
    elif model_name == 'resnet-34':
        model = models.resnet34(weights = models.ResNet34_Weights.DEFAULT)
    elif model_name == 'resnet-50':
        model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError("Invalid model name. Choose from 'resnet-18', 'resnet-34', 'resnet-50'.")

    if ft_type not in ['full', 'classifier']:
        raise ValueError("Invalid finetuning scheme. Choose from 'full' or 'classifier'.")
    
    if ft_type == 'classifier':
        print(f'\nFreezing non-classifier head weights ....\n')
        for param in model.parameters():
            param.requires_grad = False  # Freeze all layers
        # Unfreeze the classifier (fully connected layer)
        for param in model.fc.parameters():
            param.requires_grad = True

    return model

def load_clip(num_patches, num_classes, model_name):
    classifier = None
    if num_patches == 16:
        model, processor = clip.load('ViT-B/16', device='cuda:0')
    elif num_patches == 32:
        model, processor = clip.load('ViT-B/32', device='cuda:0')
    
    if model_name == "clip-classifier":
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Linear(model.visual.output_dim, num_classes)

    return model, processor, classifier

def load_vit(model_name, ft_type):
    model_size = model_name.split("_")[1].lower()
    num_patches = model_name.split("_")[2]

    if model_size not in ["small", "base", "large"]:
        raise ValueError("Invalid model size.")
    if num_patches not in ["16", "32"]:
        raise ValueError("Invalid number of patches.")

    name = f"vit_{model_size}_patch{num_patches}_224"
    model = timm.create_model(name, pretrained=True)

    if ft_type not in ["full", "classifier"]:
        raise ValueError("Invalid finetuning scheme. Choose from 'full' or 'classifier'.")

    # free layers except classifier head if ft_type='classifier'
    if ft_type == "classifier":
        for param in model.parameters():
            param.requires_grad = False  # Freeze all layers
        # Unfreeze the classifier
        for param in model.head.parameters():
            param.requires_grad = True

    return model
