from torch.utils import data
import torch
import os
from PIL import Image
class EvalDataset(data.Dataset):
    def __init__(self, img_root, label_root):
        lst_label = sorted(os.listdir(label_root))
        lst_pred = sorted(os.listdir(img_root))
        lst = []
        label_expand = lst_label[0].split('.')[-1]
        pred_expand = lst_pred[0].split('.')[-1]
        for name in lst_label:
            if label_expand !=pred_expand:
                name = name.replace(label_expand,pred_expand)
            if name in lst_pred:
                lst.append(name)

        self.lst = lst
        self.image_path = list(map(lambda x: os.path.join(img_root, x), lst_pred))
        self.label_path = list(map(lambda x: os.path.join(label_root, x), lst_label))

    def __getitem__(self, item):
        pred = Image.open(self.image_path[item]).convert('L')
        gt = Image.open(self.label_path[item]).convert('L')
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        img_name = self.lst[item]

        return pred, gt, img_name

    def __len__(self):
        return len(self.image_path)
