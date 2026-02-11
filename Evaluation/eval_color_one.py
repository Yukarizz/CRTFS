import os
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
from kornia import color
def angle(a, b):
    return F.cosine_similarity(a, b, dim=0).unsqueeze(1).mean()
transform = transforms.Compose([
    transforms.ToTensor()
])
target_path = r'F:\ColorFuse\focus\local\37\Vi.png'
target_img = Image.open(target_path)
target_img = transform(target_img)

fuse_path = r'F:\ColorFuse\focus\local\37\U2Fusion.png'
fuse_img = Image.open(fuse_path)
fuse_img = transform(fuse_img)
print(1-angle(color.rgb_to_xyz(fuse_img), color.rgb_to_xyz(target_img)))
print(1-angle((fuse_img), (target_img)))