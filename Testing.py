import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from dataset import get_loader
import transforms as trans
from torchvision import transforms
import time
from Models.crtfs_net import CRTFS
from torch.utils import data
import numpy as np
import os
from tqdm import tqdm
from kornia import color
def test_net(args):

    cudnn.benchmark = True

    net = CRTFS(args)
    net.cuda()
    net.eval()
    mean_torch = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]).astype('float32')).cuda()
    std_torch = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]).astype('float32')).cuda()
    # load model (multi-gpu)
    model_path = args.test_model_name
    state_dict = torch.load(model_path)
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    
    # load params
    net.load_state_dict(new_state_dict)
    total_params = sum(p.numel() for p in net.parameters())
    print('Model loaded from {}'.format(model_path))

    # load model
    # net.load_state_dict(torch.load(args.test_model_dir))
    # model_dict = net.state_dict()
    # print('Model loaded from {}'.format(args.test_model_dir))

    test_paths = args.test_paths.split('+')
    for test_dir_img in test_paths:

        test_dataset = get_loader(test_dir_img, args.data_root, args.img_size, mode='test')

        test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1)
        print('''
                   Starting testing:
                       dataset: {}
                       Testing size: {}
                   '''.format(test_dir_img.split('/')[0], len(test_loader.dataset)))
        tqdms = tqdm(test_loader)
        time_list = []
        with torch.no_grad():
            for data_batch in tqdms:
                images, depths, image_w, image_h, image_path = data_batch
                images, depths = Variable(images.cuda()), Variable(depths.cuda())
                dnorm_images = images * std_torch + mean_torch
                ycbcr_img = color.rgb_to_ycbcr(dnorm_images)

                starts = time.time()
                save_heat_path = './preds/VT5000/heatmap/' + image_path[0].split('/')[-1]
                outputs_saliency, outputs_contour, y_f, cbcr = net.fusion(images, depths, ycbcr_img,None)
                ends = time.time()
                time_use = ends - starts
                time_list.append(time_use)
                fused_rgb = cbcr * y_f
                # fused_rgb = torch.cat([y_f, cbcr[:, 0, ...].unsqueeze(1), cbcr[:, 1, ...].unsqueeze(1)], dim=1)
                # fused_rgb = color.ycbcr_to_rgb(fused_rgb)
                # fused_rgb = torch.clamp(fused_rgb, 0, 1)
                mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency

                image_w, image_h = int(image_w[0]), int(image_h[0])

                output_s = F.sigmoid(mask_1_1)
                # output_s = 0 * (output_s < 0.5) + 1. * (output_s >= 0.5)
                output_s = output_s.data.cpu().squeeze(0)

                transform = trans.Compose([
                    transforms.ToPILImage(),
                    trans.Scale((image_w, image_h))
                ])
                output_s = transform(output_s)
                fused_rgb = transform(fused_rgb.squeeze(0).cpu())
                dataset = test_dir_img.split('/')[0]
                filename = image_path[0].split('/')[-1].split('.')[0]

                # save saliency maps
                save_test_path = args.save_test_path_root + dataset + '/RGBD_VST/'
                save_fusion_path = save_test_path.replace('/RGBD_VST/', '/fusion/')
                if not os.path.exists(save_test_path):
                    os.makedirs(save_test_path)
                if not os.path.exists(save_fusion_path):
                    os.makedirs(save_fusion_path)
                output_s.save(os.path.join(save_test_path, filename + '.png'))
                fused_rgb.save(os.path.join(save_fusion_path, filename + '.png'))
            print('dataset:{}, cost:{}'.format(test_dir_img.split('/')[0], np.mean(time_list)*1000))






