import os
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
from kornia import color
import torch
transform = transforms.Compose([
    transforms.ToTensor()
])
def angle(a, b):
    return F.cosine_similarity(a, b, dim=0).unsqueeze(1).mean()
def load_image_pairs(folder1, folder2):
    filenames1 = sorted(os.listdir(folder1))
    filenames2 = sorted(os.listdir(folder2))

    for filename1, filename2 in zip(filenames1, filenames2):
        img1 = Image.open(os.path.join(folder1, filename1))
        img2 = Image.open(os.path.join(folder2, filename2))

        if img1 is not None and img2 is not None:
            tensor_img1 = transform(img1)
            tensor_img2 = transform(img2)
            yield color.rgb_to_xyz(tensor_img1), color.rgb_to_xyz(tensor_img2)

folder1 = r'F:\ColorFuse\Ablation_fusion\WO_SOD' # 第1个文件夹路径
folder2 = r'F:\pythonproject\VST-main\RGBD_VST\data\LLVIP\vi' # 第2个文件夹路径
color_angle = []
for fused_rgb, vi_image in load_image_pairs(folder1, folder2):
    color_angle.append ((1 - angle((fused_rgb), (vi_image))).item())
color_angle = torch.tensor(color_angle)
print(color_angle.mean())
print(color_angle.std())