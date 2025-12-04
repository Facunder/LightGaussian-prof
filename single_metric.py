import torch
from PIL import Image
import torchvision.transforms.functional as tf
from utils.image_utils import psnr
from utils.loss_utils import ssim
from argparse import ArgumentParser


def compare_images(render_path, gt_path):
    """
    Compute PSNR and SSIM between two images.
    """
    # Load images
    render = Image.open(render_path).convert("RGB")
    gt = Image.open(gt_path).convert("RGB")

    # Convert to tensor and move to GPU
    render_tensor = tf.to_tensor(render).unsqueeze(0).cuda()
    gt_tensor = tf.to_tensor(gt).unsqueeze(0).cuda()

    # Compute metrics
    psnr_val = psnr(render_tensor, gt_tensor)
    ssim_val = ssim(render_tensor, gt_tensor)

    # Convert Tensor → float for printing
    psnr_val = psnr_val.item() if isinstance(psnr_val, torch.Tensor) else psnr_val
    ssim_val = ssim_val.item() if isinstance(ssim_val, torch.Tensor) else ssim_val

    print(f"\nImage Comparison:")
    print(f"  Render : {render_path}")
    print(f"  GT     : {gt_path}")
    print(f"  PSNR   : {psnr_val:.6f}")
    print(f"  SSIM   : {ssim_val:.6f}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Compare two images using PSNR and SSIM")
    parser.add_argument("--render", "-r", required=True, type=str, help="Path to rendered image")
    parser.add_argument("--gt", "-g", required=True, type=str, help="Path to ground truth image")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    compare_images(args.render, args.gt)
