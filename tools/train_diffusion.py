import os
import argparse
from utils.adni_slice_dataset import ADNISliceDataset
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from generative.networks.nets import DiffusionModelUNet
from monai.losses import PerceptualLoss
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer
from torch.utils.tensorboard import SummaryWriter
from monai.utils import set_determinism
from models.sononet import SonoNet16
from torch.utils.data import random_split

set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def percentile(x, q):
    k = max(int(round(q / 100. * x.numel() / x.shape[0])), 1)
    topk, _ = torch.kthvalue(x.flatten(2), k, dim=-1)
    return topk.view(x.shape[0], 1, 1, 1)          # (B,1,1,1)


def train(args):
    dataset = ADNISliceDataset(
        npy_path='/DataRead/ksoh/js_ws_data/total_dat.npy',
        labels_path='/DataRead/ksoh/js_ws_data/labels.npy'
    )

    total_len = len(dataset)
    train_len = int(0.9 * total_len)
    test_len = total_len - train_len

    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)

    # Diffusion generator model
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(32, 64, 128, 256),
        attention_levels=(False, False, True, True),
        num_res_blocks=2,
        norm_num_groups=8,
        with_conditioning=True,  # cross-attention 활성화
        cross_attention_dim=3  # 타겟 클래스 one-hot 벡터(3)
    ).to(DEVICE)

    scheduler = DDPMScheduler(
        prediction_type="epsilon",
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        beta_start=0.0015,
        beta_end=0.02
    )
    inferer = DiffusionInferer(scheduler=scheduler)

    # Pretrained SonoNet-16 classifier
    classifier = SonoNet16(in_channels=1, num_classes=3).to(DEVICE)
    classifier_path = "/home/chsong/project/brain_cf_map/checkpoints/sononet_best_ckpt.pth"
    if os.path.exists(classifier_path):
        classifier.load_state_dict(torch.load(classifier_path))
        print("Loaded classifier from checkpoint", classifier_path, flush=True)
    else:
        print("Falied Loading classifier from checkpoint", classifier_path, flush=True)
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False

    # Losses
    perceptual_loss_fn = PerceptualLoss(
        spatial_dims=2, network_type="resnet50", pretrained=True
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    best_loss, wait, patience = float("inf"), 0, 7
    global_step = 0

    for epoch in range(args.n_epochs):
        model.train()
        epoch_total_losses = []
        mse_losses = []
        cls_losses = []
        cf_map_losses = []
        perc_losses = []
        tv_losses = []

        # schedule for weights
        # if epoch < 10:
        #     cf_perceptual_weight = 0
        #     tv_weight = 0
        # elif epoch < 20:
        #     cf_perceptual_weight = 0.000005
        #     tv_weight = 0.000005
        # elif epoch < 30:
        #     cf_perceptual_weight = 0.0005
        #     tv_weight = 0.00005
        # else:
        #     cf_perceptual_weight = 0.001
        #     tv_weight = 0.0005

        if epoch < 10:
            cf_perceptual_weight = 0
            tv_weight = 0
        elif epoch < 20:
            cf_perceptual_weight = 0.0001
            tv_weight = 0.00001
        elif epoch < 40:
            cf_perceptual_weight = 0.0005
            tv_weight = 0.00005
        else:
            cf_perceptual_weight = 0.0015
            tv_weight = 0.0001

        mse_weight = 0.5
        cls_weight = 1.2
        cf_l1_weight = 0.01
        cf_l2_weight = 0.002

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        for step, batch in pbar:
            image = batch['image'].to(DEVICE)
            label = batch['label'].to(DEVICE)
            num_classes = 3

            # target condition
            target_label = torch.stack([
                torch.tensor(np.random.choice([c for c in range(num_classes) if c != l.item()]))
                for l in label
            ]).to(DEVICE)
            target_condition = F.one_hot(target_label, num_classes=num_classes).float()
            target_condition = target_condition.unsqueeze(1)

            # ---- iterative sampling ----
            n_train_steps = 5
            x_t = image.clone()

            for i in range(n_train_steps):
                t = torch.randint(0, scheduler.num_train_timesteps, (image.shape[0],), device=DEVICE).long()
                noise = torch.randn_like(x_t)
                noise_pred = inferer(
                    inputs=x_t,
                    diffusion_model=model,
                    noise=noise,
                    timesteps=t,
                    condition=target_condition,
                    mode='crossattn',
                )

                x_t_list = []
                for b in range(x_t.shape[0]):
                    t_b = t[b].item()
                    x_t_b, _ = scheduler.step(noise_pred[b:b + 1], t_b, x_t[b:b + 1])
                    x_t_list.append(x_t_b)
                x_t = torch.cat(x_t_list, dim=0)

                # 마지막만 gradient 유지
                if i < n_train_steps - 1:
                    x_t = x_t.detach()

            x0_norm = torch.sigmoid(x_t)

            # percentile matching
            p_low, p_high = 10.0, 90.0
            img_lo = percentile(image, p_low)
            img_hi = percentile(image, p_high)
            pred_lo = percentile(x0_norm, p_low)
            pred_hi = percentile(x0_norm, p_high)
            scale = (img_hi - img_lo) / (pred_hi - pred_lo + 1e-6)
            cf_image = (x0_norm - pred_lo) * scale + img_lo

            μ_img, σ_img = image.mean([2, 3], keepdim=True), image.std([2, 3], keepdim=True) + 1e-6
            μ_cf, σ_cf = cf_image.mean([2, 3], keepdim=True), cf_image.std([2, 3], keepdim=True) + 1e-6
            cf_image = (cf_image - μ_cf) * (σ_img / σ_cf) + μ_img
            cf_image = torch.clamp(cf_image, 0., 1.)

            # === perceptual & tv 강화 ===
            cf_image_3c = cf_image.repeat(1, 3, 1, 1)
            image_3c = image.repeat(1, 3, 1, 1)
            loss_perc = perceptual_loss_fn(cf_image_3c, image_3c)
            tv_loss = (torch.mean(torch.abs(cf_image[:, :, 1:, :] - cf_image[:, :, :-1, :])) +
                       torch.mean(torch.abs(cf_image[:, :, :, 1:] - cf_image[:, :, :, :-1])))

            # === classifier loss ===
            logits = classifier(cf_image)
            loss_cls = F.cross_entropy(logits, target_label.long())

            # === cf_map loss ===
            cf_map = cf_image - image
            loss_cf_map = cf_l1_weight * torch.mean(torch.abs(cf_map)) + cf_l2_weight * torch.mean(cf_map ** 2)

            # === total loss ===
            total_loss = (
                    mse_weight * F.mse_loss(cf_image, image) +
                    cls_weight * loss_cls +
                    loss_cf_map +
                    cf_perceptual_weight * loss_perc +
                    tv_weight * tv_loss
            )

            # === record losses ===
            mse_losses.append(F.mse_loss(cf_image, image).item())
            cls_losses.append(loss_cls.item())
            cf_map_losses.append(loss_cf_map.item())
            perc_losses.append(cf_perceptual_weight * loss_perc.item())
            tv_losses.append(tv_weight * tv_loss.item())
            epoch_total_losses.append(total_loss.item())

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            pbar.set_postfix({'batch_loss': total_loss.item()})

        # === Print epoch summary ===
        print(
            f"Epoch {epoch + 1} | Total: {np.mean(epoch_total_losses):.4f} | "
            f"MSE: {np.mean(mse_losses):.4f} | CLS: {np.mean(cls_losses):.4f} | "
            f"CF_map: {np.mean(cf_map_losses):.4f} | Perceptual: {np.mean(perc_losses):.4f} | "
            f"TV: {np.mean(tv_losses):.4f}"
        )

        avg_loss = np.mean(epoch_total_losses)

        # Early stopping
        if avg_loss < best_loss - 1e-4:
            best_loss, wait = avg_loss, 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"unet_best_ckpt_2.pth"))
            print(f" Saved best model at epoch {epoch}")
        else:
            wait += 1
            if wait >= patience:
                torch.save(model.state_dict(), os.path.join(args.output_dir, f"unet_ckpt_2.pth"))
                print(f" Early stopping at epoch {epoch}")
                return
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"unet_ckpt_2.pth"))

    print("Done Training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
