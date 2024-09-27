import argparse
import time 
import csv
import matplotlib.pyplot as plt 
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import *
from data import *
from utils import *

def train_step(model, dataloader, criterion, optimizer, device):
    '''Train for one epoch'''
    
    model.train()

    train_loss = 0.0
    train_acc = 0.0

    for i, data in enumerate(dataloader):

        X, y = data[0].to(device), data[1].to(device)
        
        logits = model(X)
        loss = criterion(logits, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred = torch.argmax(logits.detach(), dim=1)
        train_acc += (y_pred == y).sum().item() / len(y)

        # Print dynamic progress on the same line using \r
        print(f'\rTraining: [{i+1}/{len(dataloader)}] '
              f'Loss: {train_loss / (i + 1):.4f} '
              f'Acc: {train_acc / (i + 1):.4f}', end='')

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    
    # Move to the next line after the loop is done
    print()  
    
    return train_loss, train_acc

@torch.inference_mode()
def eval_step(model, dataloader, criterion, device):
    '''Evaluate the model'''
    
    model.eval()

    eval_loss = 0.0
    eval_acc = 0.0

    for i, data in enumerate(dataloader):

        X, y = data[0].to(device), data[1].to(device)
        
        logits = model(X)
        loss = criterion(logits, y)
        eval_loss += loss.item()

        y_pred = torch.argmax(logits.detach(), dim=1)
        eval_acc += (y_pred == y).sum().item() / len(y)

        # Print dynamic progress on the same line using \r
        print(f'\rEvaluation: [{i+1}/{len(dataloader)}] '
              f'Loss: {eval_loss / (i + 1):.4f} '
              f'Acc: {eval_acc / (i + 1):.4f}', end='')

    eval_loss = eval_loss / len(dataloader)
    eval_acc = eval_acc / len(dataloader)
    
    # Move to the next line after the loop is done
    print()  
    
    return eval_loss, eval_acc

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Finetune a pretrained ResNet!')
    parser.add_argument('--model_name', type=str, default='resnet-34', help='Choose from [resnet-34, vit or clip]. Default is resnet-34.')
    parser.add_argument('--save_model', type=bool, help='Specify whether the trained model must be saved or not. Will be saved in {output_dir}/models.', default=True)
    parser.add_argument('--out_dir', type=str, default='./output', help='Path to the directory where training log and model will be saved.')
    parser.add_argument('--train_path', type=str, help='Path to training data.')
    parser.add_argument('--val_path', type=str, help='Path to validation data.')
    parser.add_argument('--lr', type=float, help='Learning Rate. Default is 1e-4', default=1e-4)
    parser.add_argument('--batch_size', type=int, help='Batch size. Default is 32.', default=32)
    parser.add_argument('--epochs', type=int, help='Number of fine-tuning epochs. Default is 15', default=15)
    
    args = parser.parse_args()

    set_seed()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size = args.batch_size
    num_workers = 2

    os.makedirs(f'{args.out_dir}/model', exist_ok=True)
    os.makedirs(f'{args.out_dir}/log', exist_ok=True)

    train_ds, test_ds = get_custom_data(args.train_path, args.val_path)

    train_dl = get_dataloader(train_ds, batch_size, True, num_workers)
    test_dl = get_dataloader(test_ds, batch_size, False, num_workers)

    num_classes = len(train_ds.classes)

    model = load_resnet_ft(args.model_name, 'classifier')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    print(f'\nFinetuning {args.model_name} with {count_parameters(model) * 1e-6:.3f}M params for {args.epochs} epochs on {device}...\n')

    best_loss = float('inf')
    bar_format = '{l_bar}{bar} | Epoch: {n_fmt}/{total_fmt} | Time: {elapsed} < {remaining} | {rate_fmt}'

    with open(os.path.join(args.out_dir, 'log/run.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])

        for epoch in tqdm(range(args.epochs), desc="Epochs", bar_format=bar_format, leave=True):
            start_time = time.time()  # Track the start time of the epoch

            train_loss, train_acc = train_step(model, train_dl, criterion, optimizer, device)
            test_loss, test_acc = eval_step(model, test_dl, criterion, device)
            
            if test_loss < best_loss:
                best_loss = test_loss
                if args.save_model:
                    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model/best.pth'))

            writer.writerow([epoch + 1, train_loss, train_acc, test_loss, test_acc])
            
            # Calculate epoch duration
            epoch_duration = time.time() - start_time
            
            # Use tqdm.write to print epoch summary with duration
            tqdm.write(f"============ Epoch {epoch + 1} --> Train Acc: {train_acc:.4f} || Test Acc: {test_acc:.4f} || Time: {epoch_duration:.2f} s ============\n")
            
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











        