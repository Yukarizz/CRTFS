import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from Evaluation.seg_eval import *
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision.utils import save_image
from dataset import get_loader
import math
from Models.crtfs_net import CRTFS
import os
import numpy as np
from kornia import color
from kornia.losses import binary_focal_loss_with_logits
from losses import *
from contextual.contextual import ContextualLoss
from tqdm import tqdm
def save_loss(save_dir, whole_iter_num, epoch_total_loss, epoch_loss, epoch):
    fh = open(save_dir, 'a')
    epoch_total_loss = str(epoch_total_loss)
    epoch_loss = str(epoch_loss)
    fh.write('until_' + str(epoch) + '_run_iter_num' + str(whole_iter_num) + '\n')
    fh.write(str(epoch) + '_epoch_total_loss' + epoch_total_loss + '\n')
    fh.write(str(epoch) + '_epoch_loss' + epoch_loss + '\n')
    fh.write('\n')
    fh.close()


def adjust_learning_rate(optimizer, decay_rate=.1):
    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        print('before lr: ', param_group['lr'])
        param_group['lr'] = param_group['lr'] * decay_rate
        print('after lr: ', param_group['lr'])
    return optimizer


def save_lr(save_dir, optimizer):
    update_lr_group = optimizer.param_groups[0]
    fh = open(save_dir, 'a')
    fh.write('encode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('decode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('\n')
    fh.close()


def train_net(num_gpus, args):

    # mp.spawn(main, nprocs=num_gpus, args=(num_gpus, args))

    if num_gpus <= 1:
        main(0, 1, args)   # local_rank=0, world_size=1
    else:
        mp.spawn(main, nprocs=num_gpus, args=(num_gpus, args))


def main(local_rank, num_gpus, args):

    cudnn.benchmark = True
    is_ddp = num_gpus > 1

    if is_ddp:
        dist.init_process_group(
            backend="gloo",
            init_method=args.init_method,   # 多卡时才用它
            world_size=num_gpus,
            rank=local_rank
        )

    net = CRTFS(args)

    # net.train().cuda()
    torch.cuda.set_device(local_rank)

    if is_ddp:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
    test_dataset = get_loader('VT5000', args.data_root, args.img_size, mode='eval')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1)
    if args.resume:
        print("Load checkpoint from %s"%args.resume)
        state_dict = torch.load(args.resume)
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        net.load_state_dict(new_state_dict)
        best_mIou = 0.0
    else:
        best_mIou = 0.0
    net.train()
    net.cuda()
    mean_torch = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]).astype('float32')).cuda()
    std_torch = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]).astype('float32')).cuda()
    # net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    # net = torch.nn.parallel.DistributedDataParallel(
    #     net,
    #     device_ids=[local_rank],
    #     output_device=local_rank,
    #     find_unused_parameters=True)
    contextual = ContextualLoss().cuda()
    base_params = [params for name, params in net.named_parameters() if ("backbone" in name)]
    other_params = [params for name, params in net.named_parameters() if ("backbone" not in name)]

    optimizer = optim.Adam([{'params': base_params, 'lr': args.lr * 0.1},
                            {'params': other_params, 'lr': args.lr}])
    train_dataset = get_loader(args.trainset, args.data_root, args.img_size, mode='train')

    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=num_gpus,
        rank=local_rank,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=6,
                                               pin_memory=True,
                                               sampler=sampler,
                                               drop_last=True,
                                               )

    print('''
        Starting training:
            Train steps: {}
            Batch size: {}
            Learning rate: {}
            Training size: {}
        '''.format(args.train_steps, args.batch_size, args.lr, len(train_loader.dataset)))

    N_train = len(train_loader) * args.batch_size

    loss_weights = [1, 0.8, 0.8, 0.5, 0.5, 0.5]
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)
    criteria_sod = SODLoss().cuda()

    criterion = nn.BCEWithLogitsLoss()
    whole_iter_num = 0
    iter_num = math.ceil(len(train_loader.dataset) / args.batch_size)
    for epoch in range(args.epochs):
        train_dataset.shuffle()
        test_dataset.shuffle()
        print('Starting epoch {}/{}.'.format(epoch + 1, args.epochs))
        print('epoch:{0}-------lr:{1}'.format(epoch + 1, args.lr))
        epoch_total_loss = 0
        epoch_loss = 0
        seg_metric = SegmentationMetric(2, device='cuda')
        for i, data_batch in enumerate(train_loader):
            if (i + 1) > iter_num: break

            images, depths, label_224, label_14, label_28, label_56, label_112, \
            contour_224, contour_14, contour_28, contour_56, contour_112 = data_batch

            images, depths, label_224, contour_224 = Variable(images.cuda(local_rank, non_blocking=True)), \
                                        Variable(depths.cuda(local_rank, non_blocking=True)), \
                                        Variable(label_224.cuda(local_rank, non_blocking=True)),  \
                                        Variable(contour_224.cuda(local_rank, non_blocking=True))

            label_14, label_28, label_56, label_112 = Variable(label_14.cuda()), Variable(label_28.cuda()),\
                                                      Variable(label_56.cuda()), Variable(label_112.cuda())

            contour_14, contour_28, contour_56, contour_112 = Variable(contour_14.cuda()), \
                                                                                      Variable(contour_28.cuda()), \
                                                      Variable(contour_56.cuda()), Variable(contour_112.cuda())

            dnorm_images = images * std_torch + mean_torch
            dnorm_depths = depths * std_torch + mean_torch
            ycbcr_img = color.rgb_to_ycbcr(dnorm_images)
            outputs_saliency, outputs_contour, y_f, cbcr, rec_grad = net(images, depths, ycbcr_img)
            # fused_rgb = torch.cat([y_f, cbcr[:,0,...].unsqueeze(1), cbcr[:,1,...].unsqueeze(1)],dim=1)
            # fused_rgb = color.ycbcr_to_rgb(fused_rgb)
            fused_rgb = cbcr * y_f
            fused_ycbcr = color.rgb_to_ycbcr(fused_rgb)
            fused_ycbcr = torch.clamp(fused_ycbcr,0,1)
            fusion_loss = 2*Fusion_loss(dnorm_depths[:,0,...].unsqueeze(1), ycbcr_img[:,0,...].unsqueeze(1), fused_ycbcr[:,0,...].unsqueeze(1), device='cuda')
            contextual_loss = contextual(fused_rgb,dnorm_images)
            color_anlge_loss = 5*angle(color.rgb_to_xyz(fused_rgb),color.rgb_to_xyz(dnorm_images))
            color_loss = contextual_loss + color_anlge_loss

            rec_loss = Re_loss(dnorm_depths[:,0,...].unsqueeze(1),ycbcr_img[:,0,...].unsqueeze(1),rec_grad)
            mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency
            cont_1_16, cont_1_8, cont_1_4, cont_1_1 = outputs_contour
            # loss
            loss5 = criteria_sod(mask_1_16, label_14)
            loss4 = criteria_sod(mask_1_8, label_28)
            loss3 = criteria_sod(mask_1_4, label_56)
            loss1 = criteria_sod(mask_1_1, label_224)

            # contour loss
            c_loss5 = boundary_loss(cont_1_16, contour_14)
            c_loss4 = boundary_loss(cont_1_8, contour_28)
            c_loss3 = boundary_loss(cont_1_4, contour_56)
            c_loss1 = boundary_loss(cont_1_1, contour_224)

            img_total_loss = 5*(loss_weights[0] * loss1 + loss_weights[2] * loss3 + loss_weights[3] * loss4 + loss_weights[4] * loss5)
            contour_total_loss = loss_weights[0] * c_loss1 + loss_weights[2] * c_loss3 + loss_weights[3] * c_loss4 + loss_weights[4] * c_loss5

            total_loss = img_total_loss + contour_total_loss + fusion_loss + color_loss + rec_loss

            epoch_total_loss += total_loss.cpu().data.item()
            epoch_loss += loss1.cpu().data.item()

            print(
                'whole_iter_num: {0} --- {1:.4f} --- total_loss: {2:.6f} --- saliency loss: {3:.6f} --- fusion loss: {4:.6f} --- contextual loss: {5:.6f} --- rec loss: {6:.6f} --- color:{7:.6f}'.format(
                    (whole_iter_num + 1),
                    (i + 1) * args.batch_size / N_train,
                    total_loss.item(),
                    img_total_loss.item(),
                    fusion_loss.item(),
                    contextual_loss.item(),
                    rec_loss.item(),
                    color_anlge_loss.item()
                    ))

            optimizer.zero_grad()

            total_loss.backward()

            optimizer.step()
            whole_iter_num += 1



            if whole_iter_num == args.train_steps:
                return 0

            if whole_iter_num == args.stepvalue1 or whole_iter_num == args.stepvalue2:
                optimizer = adjust_learning_rate(optimizer, decay_rate=args.lr_decay_gamma)
                save_dir = './loss.txt'
                save_lr(save_dir, optimizer)
                print('have updated lr!!')
        # if (local_rank == 0):
        #     torch.save(net.state_dict(),
        #                args.save_model_dir + 'RGBD_VST.pth')
        output_s = torch.nn.functional.sigmoid(mask_1_1)
        outputs_c = torch.nn.functional.sigmoid(cont_1_1)
        save_image(output_s,"./training_output/epoch%s.png"%epoch)
        save_image(outputs_c, "./training_output/epoch%s_contour.png" % epoch)
        save_image(dnorm_images, "./training_output/epoch%s_vi.png" % epoch)
        save_image(torch.clamp(fused_rgb,0,1), "./training_output/epoch%s_fused.png" % epoch)
        # rec_grad = rec_grad.view(-1, 224, 224).unsqueeze_(1)
        # save_image(rec_grad, "./training_output/epoch%s_grad.png" % epoch, nrow=8, normalize=True)

        print('Epoch finished ! Loss: {}'.format(epoch_total_loss / iter_num))
        save_lossdir = './loss.txt'
        save_loss(save_lossdir, whole_iter_num, epoch_total_loss / iter_num, epoch_loss/iter_num, epoch+1)
        if epoch>=0:
            print("start eval:")
            test_mIoU = multi_task_tester(test_loader, net, 'cuda')
            print("test_mIoU: %s" % test_mIoU)
            print("best_mIoU: %s" % best_mIou)
            if test_mIoU > best_mIou:
                best_mIou = test_mIoU
                torch.save(net.state_dict(),args.save_model_dir + 'RGBD_VST_%s_%.2f.pth'%(epoch,best_mIou))
                print("best mIoU update: %s"%test_mIoU)

def multi_task_tester(test_loader, multi_task_model, device):
    net = (multi_task_model.module if hasattr(multi_task_model, "module") else multi_task_model).eval()
    test_bar= tqdm(test_loader)
    seg_metric = SegmentationMetric(2, device=device)
    lb_ignore = [255]
    mean_torch = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]).astype('float32')).cuda()
    std_torch = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]).astype('float32')).cuda()
    count = 0
    with torch.no_grad():  # operations inside don't track history
        for data_batch in test_bar:
            img_vi, img_ir, image_w, image_h, image_path,label = data_batch
            img_ir = img_ir.to(device)
            img_vi = img_vi.to(device)
            label = label.to(device)
            dnorm_images = img_vi * std_torch + mean_torch
            ycbcr_img = color.rgb_to_ycbcr(dnorm_images)
            outputs_saliency, outputs_contour, y_f, cbcr = net.fusion(img_vi, img_ir, ycbcr_img)
            mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency
            image_w, image_h = int(image_w[0]), int(image_h[0])
            output_s = F.sigmoid(mask_1_1)
            output_s = F.interpolate(output_s,[image_h,image_w])
            output_s = 0 * (output_s < 0.5) + 1 * (output_s >= 0.5)
            seg_metric.addBatch(output_s.to(torch.int8), label.to(torch.int8), lb_ignore)
            count +=1
            if count==500:
                break
    mIoU = np.array(seg_metric.meanIntersectionOverUnion().item())
    return mIoU