import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.utils import set_determinism
from monai.losses import GlobalMutualInformationLoss
from torch.utils.data import DataLoader, random_split

from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler
from generative.inferers import DiffusionInferer
from utils.adni_slice_dataset import ADNISliceDataset

set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------- Utility functions -----------
def percentile(x, q):
    k = max(int(round(q / 100. * x.numel() / x.shape[0])), 1)
    topk, _ = torch.kthvalue(x.flatten(2), k, dim=-1)
    return topk.view(x.shape[0], 1, 1, 1)

def compute_NCC(x, y):
    x = x - x.mean()
    y = y - y.mean()
    return torch.sum(x * y) / (torch.sqrt(torch.sum(x**2)) * torch.sqrt(torch.sum(y**2)) + 1e-8)

def plot_and_save_grid(input_img, cf_map, cf_image, gt_map, target_img, scenario, save_dir):
    input_img = input_img.squeeze().cpu().numpy()
    cf_map = cf_map.squeeze().cpu().numpy()
    cf_image = cf_image.squeeze().cpu().numpy()
    gt_map = gt_map.squeeze().cpu().numpy()
    target_img = target_img.squeeze().cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(12,8))
    vmax_diff = max(np.abs(cf_map).max(), np.abs(gt_map).max())

    # top row
    axes[0,0].imshow(input_img, cmap='gray')
    axes[0,0].set_title("Input")
    axes[0,1].imshow(cf_map, cmap='bwr', vmin=-vmax_diff, vmax=vmax_diff)
    axes[0,1].set_title("cf_map")
    axes[0,2].imshow(cf_image, cmap='gray')
    axes[0,2].set_title("cf_image")

    # bottom row
    axes[1,0].imshow(input_img, cmap='gray')
    axes[1,0].set_title("Input")
    axes[1,1].imshow(gt_map, cmap='bwr', vmin=-vmax_diff, vmax=vmax_diff)
    axes[1,1].set_title("gt_map")
    axes[1,2].imshow(target_img, cmap='gray')
    axes[1,2].set_title("Target")

    for ax in axes.flatten():
        ax.axis('off')
    plt.suptitle(scenario)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{scenario}.png"))
    plt.close()

# def plot_and_save_grid(input_img, cf_map, cf_image, gt_map, target_img, scenario, save_dir):
#     """
#     input_img, cf_map, cf_image, gt_map, target_img : (1,H,W) or (H,W)
#     """
#     # Ensure all are (H,W)
#     input_img = input_img.squeeze().cpu().numpy()
#     cf_map = cf_map.squeeze().cpu().numpy()
#     cf_image = cf_image.squeeze().cpu().numpy()
#     gt_map = gt_map.squeeze().cpu().numpy()
#     target_img = target_img.squeeze().cpu().numpy()
#
#     # Concatenate horizontally
#     top_row = np.concatenate([input_img, cf_map, cf_image], axis=1)
#     bottom_row = np.concatenate([input_img, gt_map, target_img], axis=1)
#     # Concatenate vertically
#     grid = np.concatenate([top_row, bottom_row], axis=0)
#
#     plt.figure(figsize=(9,9))
#     plt.imshow(grid, cmap='gray')
#     plt.axis('off')
#     plt.title(scenario)
#     plt.savefig(os.path.join(save_dir, f"{scenario}.png"))
#     plt.close()


# ----------- Load dataset & models -----------
dataset = ADNISliceDataset(
    npy_path='/DataRead/ksoh/js_ws_data/total_dat.npy',
    labels_path='/DataRead/ksoh/js_ws_data/labels.npy'
)

total_len = len(dataset)
train_len = int(0.9 * total_len)
test_len = total_len - train_len
generator = torch.Generator().manual_seed(42)
_, test_dataset = random_split(dataset, [train_len, test_len], generator=generator)

# pick one sample for each CN, MCI, AD
sample_CN = next(x for x in test_dataset if x['label'].item() == 0)
sample_MCI = next(x for x in test_dataset if x['label'].item() == 1)
sample_AD = next(x for x in test_dataset if x['label'].item() == 2)

samples_dict = { "CN": sample_CN, "MCI": sample_MCI, "AD": sample_AD }

# ----------- Load model & scheduler -----------
model = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(32, 64, 128, 256),
    attention_levels=(False, False, True, True),
    num_res_blocks=2,
    norm_num_groups=8,
    with_conditioning=True,
    cross_attention_dim=3
).to(DEVICE)

ckpt_path = "/home/chsong/project/brain_cf_map/checkpoints/unet_best_ckpt.pth"
model.load_state_dict(torch.load(ckpt_path))
model.eval()

scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    schedule='scaled_linear_beta',
    beta_start=0.0015,
    beta_end=0.02
)

inferer = DiffusionInferer(scheduler=scheduler)

# ----------- Sampling function -----------
@torch.no_grad()
def generate_cf_image(input_image, target_condition):
    scheduler.set_timesteps(num_inference_steps=50)
    noise = torch.randn_like(input_image).to(DEVICE)
    for t in tqdm(scheduler.timesteps, desc="Sampling"):
        timestep = torch.tensor([t], device=DEVICE)
        noise_pred = model(
            x=noise,
            timesteps=timestep,
            context=target_condition
        )
        noise, _ = scheduler.step(noise_pred, t, noise)
    x0_norm = torch.sigmoid(noise)

    # percentile normalization + matching
    p_low, p_high = 3.0, 97.0
    img_lo = percentile(input_image, p_low)
    img_hi = percentile(input_image, p_high)
    pred_lo = percentile(x0_norm, p_low)
    pred_hi = percentile(x0_norm, p_high)
    scale = (img_hi - img_lo) / (pred_hi - pred_lo + 1e-6)
    cf_image = (x0_norm - pred_lo) * scale + img_lo

    μ_img, σ_img = input_image.mean([2,3], keepdim=True), input_image.std([2,3], keepdim=True) + 1e-6
    μ_cf, σ_cf = cf_image.mean([2,3], keepdim=True), cf_image.std([2,3], keepdim=True) + 1e-6
    cf_image = (cf_image - μ_cf) * (σ_img / σ_cf) + μ_img
    cf_image = torch.clamp(cf_image, 0., 1.)
    return cf_image

# ----------- Evaluate scenarios -----------
os.makedirs("./cf_results", exist_ok=True)
scenarios = [
    ("CN", "MCI"), ("CN", "AD"), ("MCI", "AD"),
    ("MCI", "CN"), ("AD", "CN"), ("AD", "MCI")
]

for src, tgt in scenarios:
    input_image = samples_dict[src]['image'].unsqueeze(0).to(DEVICE)
    target_image = samples_dict[tgt]['image'].unsqueeze(0).to(DEVICE)
    target_condition = F.one_hot(torch.tensor([{"CN":0,"MCI":1,"AD":2}[tgt]]), num_classes=3).float().unsqueeze(1).to(DEVICE)

    print(f"==== Sampling: {src} -> {tgt} ====")
    cf_image = generate_cf_image(input_image, target_condition)
    cf_map = cf_image - input_image
    gt_map = target_image - input_image

    ncc_score = compute_NCC(cf_map, gt_map).item()
    print(f"[{src} -> {tgt}] NCC between predicted cf_map and gt_map: {ncc_score:.4f}")

    # save grid
    plot_and_save_grid(
        input_image.squeeze(),
        cf_map.squeeze(),
        cf_image.squeeze(),
        gt_map.squeeze(),
        target_image.squeeze(),
        f"{src}_{tgt}",
        "./cf_results"
    )
