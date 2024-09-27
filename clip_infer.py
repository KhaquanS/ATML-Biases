import argparse
import time 
import csv
import matplotlib.pyplot as plt 
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from train_clip import clip_loss

from model import *
from data import *
from utils import *

@torch.inference_mode()
def eval_step_clip(model, classifier, dataloader, criterion, device, labels, model_type, name):
    '''Evaluate the model'''
    
    model.eval()

    eval_loss = 0.0
    eval_acc = 0.0
    total_samples = 0

    for i, data in enumerate(dataloader):
        
        if name in ['PACS']:
            X = data['images'].to(device)
            y = torch.squeeze(data['labels']).to(device)
        else:
            X, y = data[0].to(device), data[1].to(device)

        if model_type == "clip-vit":
            logits_per_image, logits_per_text = model(X, labels)
            preds = torch.argmax(logits_per_image, dim=1)
            loss = clip_loss(logits_per_image, logits_per_text, device)
            eval_loss += loss.item()
            
        elif model_type == "clip-classifier":
            with torch.no_grad():
                image_feature = model.encode_image(X).to(torch.float32)  # Ensure this is float32
                logits_per_image = classifier(image_feature)
                loss = criterion(logits_per_image, y)
                eval_loss += loss.item()
                preds = torch.argmax(logits_per_image.detach(), dim=1)
        
        eval_acc += (preds == y).sum().item()
        total_samples += len(y)

        # Print dynamic progress on the same line using \r
        print(f'\rEvaluation: [{i+1}/{len(dataloader)}] '
              f'Loss: {eval_loss / (i + 1):.4f} '
              f'Acc: {eval_acc / total_samples:.4f}', end='')

    eval_loss = eval_loss / len(dataloader)
    eval_acc = eval_acc / total_samples
    
    # Move to the next line after the loop is done
    print()  
    
    return eval_loss, eval_acc


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Infer Pretrained Models')
    parser.add_argument('--model_name', type=str, default='resnet-34', help='Either resnet-18 or resnet-34 or resnet-50 or clip-vit or clip-classifier. Default is resnet-34.')
    parser.add_argument('--dataset', type=str, help='One of [MNIST, CIFAR-10, CIFAR-100, PACS, SVHN]. Default is CIFAR-10.', default='CIFAR-10')
    parser.add_argument('--out_dir', type=str, help='Output directory. Default is output.', default='output')
    parser.add_argument('--num_patches', type=int, help='Number of patches for ViT. Choose from 16 and 32. Default is 32', default=32)
    parser.add_argument('--num_classes', type=int, help='Number of classification classes. 10 (default) for all datasets except cifar-100 which has 100 classes.', default=10)
    parser.add_argument('--bias_evaluation', type=bool, default=False, help="Evaluate on biases of the dataset. Set it to true True if Bias Evaluation is required.", required=False)
    parser.add_argument('--train_path', type=str, help='Path to training data. Required for bias evaluation', required=False)
    parser.add_argument('--val_path', type=str, help='Path to validation data. Required for bias evaluation', required=False)
    args = parser.parse_args()
    
    set_seed()
    batch_size = 32
    num_workers = 2
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model, processor, classifier = load_clip(args.num_patches, args.num_classes, args.model_name)
    model.to(device)

    if args.bias_evaluation == True:
        print("Bias Evaluation Set to True... Evaluating on inductive biases")
        train_ds, val_ds, class_text = get_custom_data(train_path=args.train_path, val_path=args.val_path, 
                                                        processor=processor)
        class_text = class_text.to(device)
    else:
        train_ds, test_ds, class_text = get_dataset_func(args.dataset, args.model_name, processor, args.out_dir)
        class_text = class_text.to(device)

    if classifier is not None:
        classifier.to(device)


    if args.dataset == 'PACS':
        train_dl, test_dl = get_deeplake_dataloader(train_ds, test_ds, batch_size, num_workers)
        num_classes = len(train_ds.labels.info.class_names)

    else:
        train_dl = get_dataloader(train_ds, batch_size, True, num_workers)
        test_dl = get_dataloader(test_ds, batch_size, False, num_workers)
        if args.dataset == 'SVHN':
            num_classes = 10
        else:
            num_classes = len(train_ds.classes)

    criterion = nn.CrossEntropyLoss()
    
    if args.bias_evaluation == True:
        eval_loss_nonsl, eval_acc_nonsl = eval_step_clip(model, classifier, train_dl, criterion, device, class_text, args.modal_name, args.dataset)
        print(f'============ Eval Acc on Non Stylized Dataset: {eval_acc_nonsl*100:.4f}%  ============')
        print(f'============ Eval Loss on Non Stylized Dataset: {eval_loss_nonsl:.4f}  ============')
    eval_loss, eval_acc = eval_step_clip(model, classifier, test_dl, criterion, device, class_text, args.model_name, args.dataset)

    print(f'============ Eval Acc on {args.dataset}: {eval_acc*100:.4f}%  ============')
    print(f'============ Eval Loss on {args.dataset}: {eval_loss:.4f}  ============')
    

