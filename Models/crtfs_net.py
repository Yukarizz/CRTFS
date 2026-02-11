import torch.nn as nn
from .t2t_vit import T2t_vit_t_14
from .Transformer import Transformer
from .Transformer import token_Transformer
from .Decoder import Decoder
from kornia import color
import torch
from .PSF import *
from torchvision.models import ResNet34_Weights
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from thop import profile
import seaborn as sns
class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        # Shallow feature extraction
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3,
                                    padding=4, padding_mode='replicate')
        # Activated layers!
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU())
        # Final recon layer
        self.net1_recon = nn.Conv2d(channel, 4, kernel_size,
                                    padding=1, padding_mode='replicate')

    def forward(self, input_im):
        input_max= torch.max(input_im, dim=1, keepdim=True)[0]
        input_img= torch.cat((input_max, input_im), dim=1)
        feats0   = self.net1_conv0(input_img)
        featss   = self.net1_convs(feats0)
        outs     = self.net1_recon(featss)
        R        = torch.sigmoid(outs[:, 0:3, :, :])
        L        = torch.sigmoid(outs[:, 3:4, :, :])
        return R, L

class CRTFS(nn.Module):
    def __init__(self, args):
        super(CRTFS, self).__init__()

        # VST Encoder
        self.rgb_backbone = T2t_vit_t_14(pretrained=True, args=args,inchannel=3)
        self.ir_backbone = T2t_vit_t_14(pretrained=True, args=args,inchannel=3)
        self.color_backbone = T2t_vit_t_14(pretrained=False, args=args, inchannel=2)
        # VST Convertor
        self.transformer = Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)

        # VST Decoder
        self.token_trans = token_Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)
        self.decoder = Decoder(embed_dim=384, token_dim=64, depth=2, img_size=args.img_size)
        ''' ### pixel Encoder ###'''
        resnet_raw_model1 = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        resnet_raw_model2 = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        ########  Thermal pixel ENCODER  ########
        self.encoder_thermal_conv1 = Feature_extract(3, 64)
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer3 = resnet_raw_model1.layer1
        ########  RGB pixel ENCODER  ########
        self.encoder_rgb_conv1 = Feature_extract(3, 64)
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer3 = resnet_raw_model2.layer1
        ########  color ENCODER  ########
        self.encoder_color_conv1 = Feature_extract(2, 64)
        self.encoder_color_bn1 = resnet_raw_model2.bn1
        self.encoder_color_relu = resnet_raw_model2.relu
        self.encoder_color_maxpool = resnet_raw_model2.maxpool
        self.encoder_color_layer3 = resnet_raw_model2.layer1

        self.low_fuse3_1 = SDFM(64, 64)
        self.low_fuse3_2 = SDFM(64, 64)
        self.low_fuse1_1 = SDFM(32, 32)
        self.cnn2token = cnn2token(64)
        # self.saliency_guide_1_4 = cnn2token(128)
        self.saliency_guide_1_1 = SIM(norm_nc=32, label_nc=64, nhidden=32)
        self.saliency_guide_color = SIM(norm_nc=32, label_nc=32, nhidden=32)
        self.decoder_dim_rec = 32
        self.rec_decoder = DSRM(self.decoder_dim_rec, self.decoder_dim_rec)
        self.color_rec_decoder = DSRM(self.decoder_dim_rec, self.decoder_dim_rec)
        self.grad_guide = grad2token(self.decoder_dim_rec)
        self.grad_proj = nn.Sequential(
            BasicConv2d(self.decoder_dim_rec, self.decoder_dim_rec, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.decoder_dim_rec),
            nn.ReLU(inplace=True)
        )
        self.pred_grad = IFP([self.decoder_dim_rec, 2])
        # self.pred_grad = nn.Sequential(
        # nn.Conv2d(in_channels=self.decoder_dim_rec, out_channels=2, kernel_size=3, padding=1),
        # nn.ReLU()
        # )
        self.pred_fusion = IFP([self.decoder_dim_rec, 1])
        self.pred_cbcr = IFP([self.decoder_dim_rec, 3])
        ''' ### pixel Encoder ###'''

        # '''Retinex decom'''
        # self.DecomNet = DecomNet()
        # ckpt_dict = torch.load(args.retinex_ckpt)
        # self.DecomNet.load_state_dict(ckpt_dict)
        # self.DecomNet.eval()
    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                      stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        input_R = 0.299*input_R[:, 0, :, :] + 0.587*input_R[:, 1, :, :] + 0.114*input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))
    def _pixel_encode(self, image, ir, cbcr):
        rgb1, rgb2 = self.encoder_rgb_conv1(image)
        rgb2 = self.encoder_rgb_relu(self.encoder_rgb_bn1(rgb2))

        thermal1, thermal2 = self.encoder_thermal_conv1(ir)
        thermal2 = self.encoder_thermal_relu(self.encoder_thermal_bn1(thermal2))

        color1, color2 = self.encoder_color_conv1(cbcr)
        color2 = self.encoder_color_relu(self.encoder_color_bn1(color2))

        rgb3 = self.encoder_rgb_layer3(self.encoder_rgb_maxpool(rgb2))
        thermal3 = self.encoder_thermal_layer3(self.encoder_thermal_maxpool(thermal2))
        color3 = self.encoder_color_layer3(self.encoder_color_maxpool(color2))

        fused_f3 = self.low_fuse3_2(self.low_fuse3_1(rgb3, thermal3), color3)  # 1/4
        fused_f1 = self.low_fuse1_1(rgb1, thermal1)                            # 1/1
        return fused_f1, fused_f3, color1
    def _vst_tokens(self, image_224, ir_224, cbcr_224):
        rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4 = self.rgb_backbone(image_224)
        ir_fea_1_16, _, ir_fea_1_4 = self.ir_backbone(ir_224)
        color_fea_1_16, _, color_fea_1_4 = self.color_backbone(cbcr_224)

        rgb_fea_1_16, ir_fea_1_16 = self.transformer(rgb_fea_1_16, ir_fea_1_16, color_fea_1_16)

        sal_fea_1_16, fea_1_16, sal_tok, cont_fea_1_16, cont_tok = self.token_trans(rgb_fea_1_16, ir_fea_1_16)
        return (sal_fea_1_16, fea_1_16, sal_tok, cont_fea_1_16, cont_tok, rgb_fea_1_8, rgb_fea_1_4)
    def _decode_and_reconstruct(self, tokens, fused_f1, fused_f3, color1, rgb_fea_1_4, contour_guide, saliency_fea_guide=None):
        sal_fea_1_16, fea_1_16, sal_tok, cont_fea_1_16, cont_tok, rgb_fea_1_8, _ = tokens

        fused_f3_refine = self.cnn2token(fused_f3, rgb_fea_1_4)

        dec_out = self.decoder(
            sal_fea_1_16, fea_1_16, sal_tok, cont_fea_1_16, cont_tok,
            rgb_fea_1_8, rgb_fea_1_4, fused_f3_refine, contour_guide
        )

        # decoder 返回里用 outputs[2] 作为 guide
        saliency_fea_guide = dec_out[2]
        seg_outputs = dec_out[:2]  # (outputs_saliency, outputs_contour) 之类

        # 1/1 引导
        fused_f1 = fused_f1 + self.saliency_guide_1_1(fused_f1, saliency_fea_guide)
        color1 = self.saliency_guide_color(color1, fused_f1)

        rec_illu_f = self.rec_decoder(fused_f1)
        rec_color  = self.color_rec_decoder(color1)

        illu_f = self.pred_fusion(rec_illu_f)
        cbcr_refine = self.pred_cbcr(rec_color)

        return seg_outputs, illu_f, cbcr_refine, saliency_fea_guide

    def forward(self, image_Input, ir_Input, ycbcr_vi):
        B, _, H, W = image_Input.shape

        cbcr_224 = torch.cat([ycbcr_vi[:,1:2], ycbcr_vi[:,2:3]], dim=1)

        tokens = self._vst_tokens(image_Input, ir_Input, cbcr_224)

        fused_f1, fused_f3, color1 = self._pixel_encode(image_Input, ir_Input, cbcr_224)

        latent_grad = self.grad_proj(fused_f1)
        rec_grad = self.pred_grad(latent_grad)
        contour_guide = self.grad_guide(latent_grad)

        seg_outputs, illu_f, cbcr_refine, _ = self._decode_and_reconstruct(
            tokens, fused_f1, fused_f3, color1, tokens[-1], contour_guide
        )

        return seg_outputs + (illu_f, cbcr_refine, rec_grad)    
    # for test only
    def fusion(self, image_Input, ir_Input, ycbcr_vi,save_heat_path = None):
        ori_image = image_Input
        ori_ir = ir_Input
        ori_ycbcr_image = ycbcr_vi
        B,_,H,W = image_Input.shape
        image_Input = F.interpolate(image_Input,[224,224])
        ir_Input = F.interpolate(ir_Input,[224,224])
        ycbcr_vi = F.interpolate(ycbcr_vi,[224,224])
        cbcr = torch.cat([ycbcr_vi[:, 1, ...].unsqueeze(1), ycbcr_vi[:, 2, ...].unsqueeze(1)], dim=1)
        # VST Encoder
        rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4 = self.rgb_backbone(image_Input)
        ir_fea_1_16, _, ir_fea_1_4 = self.ir_backbone(ir_Input)
        color_fea_1_16, _, color_fea_1_4 = self.color_backbone(cbcr)
        # VST Convertor
        rgb_fea_1_16, ir_fea_1_16 = self.transformer(rgb_fea_1_16, ir_fea_1_16, color_fea_1_16)
        # VST Decoder
        saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens = self.token_trans(rgb_fea_1_16,
                                                                                                          ir_fea_1_16)
        rgb1, rgb2 = self.encoder_rgb_conv1(image_Input)  # (240, 320)
        rgb2 = self.encoder_rgb_bn1(rgb2)  # (240, 320)
        rgb2 = self.encoder_rgb_relu(rgb2)  # (240, 320)
        thermal1, thermal2 = self.encoder_thermal_conv1(ir_Input)  # (240, 320)
        thermal2 = self.encoder_thermal_bn1(thermal2)  # (240, 320)
        thermal2 = self.encoder_thermal_relu(thermal2)  # (240, 320)
        color1, color2 = self.encoder_color_conv1(cbcr)  # (240, 320)
        color2 = self.encoder_color_bn1(color2)  # (240, 320)
        color2 = self.encoder_color_relu(color2)  # (240, 320)
        ######################################################################
        rgb3 = self.encoder_rgb_maxpool(rgb2)  # (120, 160)
        thermal3 = self.encoder_thermal_maxpool(thermal2)  # (120, 160)
        color3 = self.encoder_color_maxpool(color2)
        rgb3 = self.encoder_rgb_layer3(rgb3)  # (120, 160)
        thermal3 = self.encoder_thermal_layer3(thermal3)  # (120, 160)
        color3 = self.encoder_color_layer3(color3)
        fused_f3 = self.low_fuse3_1(rgb3, thermal3)
        fused_f3 = self.low_fuse3_2(fused_f3, color3)  # [8, 64, 56, 56]   1/4
        fused_f3_refine = self.cnn2token(fused_f3,rgb_fea_1_4)
        fused_f1 = self.low_fuse1_1(rgb1, thermal1)

        latent_grad = self.grad_proj(fused_f1)
        contour_guide = self.grad_guide(latent_grad)
        ''''''
        outputs = self.decoder(saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens,
                               rgb_fea_1_8, rgb_fea_1_4, fused_f3_refine,contour_guide)
        ######################################################################
        saliency_fea_guide = outputs[2]  # [64 h w]
        outputs = outputs[:2]
        saliency_fea_guide = F.interpolate(saliency_fea_guide,[H//4,W//4],mode='bilinear')

        '''fuse'''
        cbcr = torch.cat([ori_ycbcr_image[:, 1, ...].unsqueeze(1), ori_ycbcr_image[:, 2, ...].unsqueeze(1)], dim=1)
        rgb1, rgb2 = self.encoder_rgb_conv1(ori_image)  # (240, 320)
        rgb2 = self.encoder_rgb_bn1(rgb2)  # (240, 320)
        rgb2 = self.encoder_rgb_relu(rgb2)  # (240, 320)
        thermal1, thermal2 = self.encoder_thermal_conv1(ori_ir)  # (240, 320)
        thermal2 = self.encoder_thermal_bn1(thermal2)  # (240, 320)
        thermal2 = self.encoder_thermal_relu(thermal2)  # (240, 320)
        color1, color2 = self.encoder_color_conv1(cbcr)  # (240, 320)
        color2 = self.encoder_color_bn1(color2)  # (240, 320)
        color2 = self.encoder_color_relu(color2)  # (240, 320)
        ######################################################################
        rgb3 = self.encoder_rgb_maxpool(rgb2)  # (120, 160)
        thermal3 = self.encoder_thermal_maxpool(thermal2)  # (120, 160)
        color3 = self.encoder_color_maxpool(color2)
        rgb3 = self.encoder_rgb_layer3(rgb3)  # (120, 160)
        thermal3 = self.encoder_thermal_layer3(thermal3)  # (120, 160)
        color3 = self.encoder_color_layer3(color3)
        fused_f3 = self.low_fuse3_1(rgb3, thermal3)
        fused_f3 = self.low_fuse3_2(fused_f3, color3)  # [8, 64, 56, 56]   1/4

        fused_f1 = self.low_fuse1_1(rgb1, thermal1)  # [8, 32, 224, 224] 1/1
        if save_heat_path is not None:
            data = saliency_fea_guide.mean(dim=1).unsqueeze(1)
            data = (data - data.min()) / (data.max() - data.min())
            data = data.transpose(0, 1).squeeze().cpu().numpy()
            plt.figure(figsize=(10, 8))
            sns.heatmap(data, cmap='jet')
            plt.axis('off')
            plt.savefig(save_heat_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        fused_f1_post = fused_f1 + self.saliency_guide_1_1(fused_f1, saliency_fea_guide)
        color1 = self.saliency_guide_color(color1, fused_f1_post)

        rec_illu_f = self.rec_decoder(fused_f1_post)
        rec_color = self.color_rec_decoder(color1)

        illu_f = self.pred_fusion(rec_illu_f)
        color_rfine = self.pred_cbcr(rec_color)


        return outputs + (illu_f, color_rfine)

