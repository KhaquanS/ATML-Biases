import argparse
import time 
import csv
import matplotlib.pyplot as plt 
import pandas as pd
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import clip

from model import *
from data import *
from utils import *

def clip_loss(logits_per_image, logits_per_text, device):
    similarity_matrix = torch.mm(logits_per_image, logits_per_text)

    # Create labels for cross-entropy
    batch_size = logits_per_image.size(0)
    labels = torch.arange(batch_size).to(device)

    # Compute the contrastive loss using cross-entropy
    loss_img_to_text = nn.CrossEntropyLoss()(similarity_matrix, labels)
    loss_text_to_image = nn.CrossEntropyLoss()(similarity_matrix.t(), labels)  # transpose for text to image

    # Average the losses
    total_loss = (loss_img_to_text + loss_text_to_image) / 2
    return total_loss


def train_step(model, classifier, dataloader, criterion, optimizer, device, labels, ft_type, name):
    '''Train for one epoch'''
    model.train()

    train_loss = 0.0
    train_acc = 0.0
    total_samples = 0

    for i, data in enumerate(dataloader):
        
        if name in ['PACS']:
            X = data['images'].to(device)
            y = torch.squeeze(data['labels']).to(device)
        else:
            X, y = data[0].to(device), data[1].to(device)

        if ft_type == 'full':
            logits_per_image, logits_per_text = model(X, labels)
            loss = clip_loss(logits_per_image, logits_per_text, device)
            preds = torch.argmax(logits_per_image, dim=1)
            train_loss += loss.item()

        else:
            with torch.no_grad():
                image_feature = model.encode_image(X).to(torch.float32)
            logits_per_image = classifier(image_feature)
            loss = criterion(logits_per_image, y)
            preds = torch.argmax(logits_per_image.detach(), dim=1)
            train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc += (preds==y).sum().item()
        total_samples +=len(y)

        # Print dynamic progress on the same line using \r
        print(f'\rTraining: [{i+1}/{len(dataloader)}] '
              f'Loss: {train_loss / (i + 1):.4f} '
              f'Acc: {train_acc/len(y) / (i + 1):.4f}', end='')

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / total_samples
    
    # Move to the next line after the loop is done
    print()  
    
    return train_loss, train_acc

@torch.inference_mode()
def eval_step(model, classifier, dataloader, criterion, device, labels, ft_type, name):
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

        if ft_type == "full":
            logits_per_image, logits_per_text = model(X, labels)
            preds = torch.argmax(logits_per_image, dim=1)
            loss = clip_loss(logits_per_image, logits_per_text, device)
            eval_loss +=loss.item()
            
        elif ft_type == "classifier":
            with torch.no_grad():
                image_feature = model.encode_image(X).to(torch.float32)
            logits_per_image = classifier(image_feature)
            loss = criterion(logits_per_image, y)
            eval_loss += loss.item()
            preds = torch.argmax(logits_per_image.detach(), dim=1)

        eval_acc += (preds==y).sum().item()
        total_samples += len(y)

        # Print dynamic progress on the same line using \r
        print(f'\rEvaluation: [{i+1}/{len(dataloader)}] '
              f'Loss: {eval_loss / (i + 1):.4f} '
              f'Acc: {eval_acc/len(y) / (i + 1):.4f}', end='')

    eval_loss = eval_loss / len(dataloader)
    eval_acc = eval_acc / total_samples
    
    # Move to the next line after the loop is done
    print()  
    
    return eval_loss, eval_acc


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Finetune a pretrained ViT based CLIP')
    parser.add_argument('--model_name', type=str, default='clip-vit', help='List of models: clip-vit, clip-classifier. Default is clip-vit.')
    parser.add_argument('--out_dir', type=str, help='Path to the directory where training log and model will be saved.')
    parser.add_argument('--dataset', type=str, help='One of [MNIST, CIFAR-10, CIFAR-100, PACS, SVHN]. Default is CIFAR-10.', default='CIFAR-10')
    parser.add_argument('--num_classes', type=int, help='Number of classification classes. 10 (default) for all datasets except cifar-100 which has 100 classes.', default=10)
    parser.add_argument('--save_model', type=bool, help='Specify whether the trained model must be saved or not. Will be saved in {output_dir}/models.', default=True)
    parser.add_argument('--lr', type=float, help='Learning Rate. Default is 1e-4', default=1e-4)
    parser.add_argument('--batch_size', type=int, help='Batch size. Default is 32.', default=32)
    parser.add_argument('--epochs', type=int, help='Number of fine-tuning epochs. Default is 15', default=15)
    parser.add_argument('--ft_type', type=str, help='Specify which parts of the model to finetune. Choose from full or classifier. Default is classifier.', default='classifier')
    parser.add_argument('--num_patches', type=int, help='Specify number of patches for the ViT in Clip. Choose from 16 and 32. Default is 32', default=32, required=False)
    args = parser.parse_args()
    

    set_seed()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size = args.batch_size
    num_workers = 2
    

    os.makedirs(f'{args.out_dir}/model', exist_ok=True)
    os.makedirs(f'{args.out_dir}/log', exist_ok=True)

    
    model, processor, classifier = load_clip(args.num_patches, args.num_classes, args.model_name)
    model.to(device)
    train_ds, test_ds, class_text = get_dataset_func(args.dataset, args.model_name, processor)
    class_text = class_text.to(device)
    if classifier is not None:
        classifier.to(device)
        
    if args.dataset == "PACS":
        train_dl, test_dl = get_deeplake_dataloader(train_ds, test_ds, batch_size, num_workers)
    else:
        train_dl = get_dataloader(train_ds, batch_size, True, num_workers)
        test_dl = get_dataloader(test_ds, batch_size, False, num_workers)

    criterion = nn.CrossEntropyLoss()

    if args.model_name =="clip-vit":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        print(f'\nFinetuning CLIP with {count_parameters(model) * 1e-6:.3f}M params for {args.epochs} epochs on {device}...\n')
    else:
        optimizer = optim.AdamW(classifier.parameters(), lr=args.lr)
        print(f'\nFinetuning CLIP classifier with {count_parameters(classifier) * 1e-6:.3f}M params for {args.epochs} epochs on {device}...\n')


    best_loss = float('inf')
    bar_format = '{l_bar}{bar} | Epoch: {n_fmt}/{total_fmt} | Time: {elapsed} < {remaining} | {rate_fmt}'

    with open(os.path.join(args.out_dir, 'log/run.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])

        for epoch in tqdm(range(args.epochs), desc="Epochs", bar_format=bar_format, leave=True):
            start_time = time.time()  # Track the start time of the epoch

            train_loss, train_acc = train_step(model, classifier, train_dl, criterion, optimizer, device, class_text, args.ft_type, args.dataset)
            test_loss, test_acc = eval_step(model, classifier, test_dl, criterion, device, class_text, args.ft_type, args.dataset)
            
            if test_loss < best_loss:
                best_loss = test_loss
                if args.save_model:
                    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model/best.pth'))

            writer.writerow([epoch + 1, train_loss, train_acc, test_loss, test_acc])
            # Calculate epoch duration
            epoch_duration = time.time() - start_time
            # Use tqdm.write to print epoch summary with duration
            tqdm.write(f"============ Epoch {epoch + 1} --> Train Acc: {train_acc*100:.4f} || Test Acc: {test_acc*100:.4f} || Time: {epoch_duration:.2f} s ============\n")
            
    # Save this model at the end of run (commented out for)
    if args.save_model:
        torch.save(model.state_dict(), os.path.join(args.out_dir, 'model/final.pth'))

    df = pd.read_csv(f'{args.out_dir}/log/run.csv')
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Train Accuracy and Test Accuracy
    ax[0].plot(df['train_acc'], label='Train Accuracy', color='orange', marker='o', linewidth=2)
    ax[0].plot(df['test_acc'], label='Test Accuracy', color='blue', marker='o', linewidth=2)
    ax[0].set_title('Train vs Test Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    ax[0].grid(True)

    # Plot Train Loss and Test Loss
    ax[1].plot(df['train_loss'], label='Train Loss', color='orange', marker='o', linewidth=2)
    ax[1].plot(df['test_loss'], label='Test Loss', color='blue', marker='o', linewidth=2)
    ax[1].set_title('Train vs Test Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig(f'{args.out_dir}/log_plot.png')
    plt.show()
