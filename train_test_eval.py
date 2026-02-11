import os
import torch
import Training
import Testing
from Evaluation import main
import argparse
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--Training', default=False, type=bool, help='Training or not')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:33111', type=str, help='init_method')
    parser.add_argument('--data_root', default=r'./data/', type=str, help='data path')#'train_data','testdata','D:\pythonproject\TransEx (2)\testdata/'
    parser.add_argument('--resume',type=str,help="path to resume ckpt", default=None)#"checkpoint\RGBD_VST_W_Color.pth"
    parser.add_argument('--train_steps', default=200000, type=int, help='train_steps')
    parser.add_argument('--img_size', default=224, type=int, help='network input size')
    
    parser.add_argument('--lr_decay_gamma', default=0.1, type=int, help='learning rate decay')
    parser.add_argument('--lr', default=1e-4, type=int, help='learning rate')
    parser.add_argument('--epochs', default=200, type=int, help='epochs')
    parser.add_argument('--batch_size', default=2, type=int, help='batch_size')
    parser.add_argument('--stepvalue1', default=20000, type=int, help='the step 1 for adjusting lr')
    parser.add_argument('--stepvalue2', default=30000, type=int, help='the step 2 for adjusting lr')
    parser.add_argument('--trainset', default='VT5000', type=str, help='Trainging set')
    parser.add_argument('--save_model_dir', default='checkpoint/', type=str, help='save model path')
    # ckpts
    parser.add_argument('--pretrained_model', default='./checkpoint/80.7_T2T_ViT_t_14.pth.tar', type=str, help='load pretrained model')
    parser.add_argument('--test_model_name', default='./checkpoint/CRTFS.pth', type=str, help='save model path')
    # parser.add_argument('--retinex_ckpt',default='./checkpoint/Decom.tar', type=str, help='retinex ckpt path')
    # test
    parser.add_argument('--Testing', default=True, type=bool, help='Testing or not')
    parser.add_argument('--save_test_path_root', default='preds/', type=str, help='save saliency maps path')
    # parser.add_argument('--test_paths', type=str, default='NJUD+NLPR+DUTLF-Depth+ReDWeb-S+STERE+SSD+SIP+RGBD135+LFSD')
    parser.add_argument('--test_paths', type=str, default='demoset')
    # evaluation
    parser.add_argument('--Evaluation', default=False, type=bool, help='Evaluation or not')
    parser.add_argument('--methods', type=str, default='CAVER', help='evaluated method name')
    parser.add_argument('--save_dir', type=str, default='./', help='path for saving result.txt')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    num_gpus = torch.cuda.device_count()
    if args.Training:
        Training.train_net(num_gpus=num_gpus, args=args)
    if args.Testing:
        Testing.test_net(args)
    if args.Evaluation:
        main.evaluate(args)