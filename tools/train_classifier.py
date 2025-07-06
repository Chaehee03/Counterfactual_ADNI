import os
import argparse
from utils.adni_slice_dataset import ADNISliceDataset
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from models.sononet import SonoNet16

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(args):
    dataset = ADNISliceDataset(
        npy_path='/DataRead/ksoh/js_ws_data/total_dat.npy',
        labels_path='/DataRead/ksoh/js_ws_data/labels.npy'
    )
    train_loader = DataLoader(dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)

    model = SonoNet16(in_channels=1, num_classes=3).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    writer = SummaryWriter(args.output_dir)
    best_loss, wait, patience = float("inf"), 0, 6
    global_step = 0

    for epoch in range(args.n_epochs):
        model.train()
        epoch_losses = []
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")

        for step, batch in pbar:
            image = batch['image'].to(DEVICE)
            label = batch['label'].to(DEVICE)

            with autocast():
                logits = model(image)
                loss = F.cross_entropy(logits, label.long())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_losses.append(loss.item())
            global_step += 1
            writer.add_scalar('train/batch_loss', loss.item(), global_step)
            pbar.set_postfix({'batch_loss': loss.item()})

        avg_loss = np.mean(epoch_losses)
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

        # Early stopping
        if avg_loss < best_loss - 1e-4:
            best_loss, wait = avg_loss, 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"sononet_best_ckpt.pth"))
            print(f" Saved best model at epoch {epoch}")
        else:
            wait += 1
            if wait >= patience:
                torch.save(model.state_dict(), os.path.join(args.output_dir, f"sononet_ckpt.pth"))
                print(f" Early stopping at epoch {epoch}")
                break
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"sononet_ckpt.pth"))

    writer.close()
    print("Done Training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./checkpoints_sononet')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
