# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.sub_modules.pillar_vfe import PillarVFE
from models.sub_modules.point_pillar_scatter import PointPillarScatter
from models.sub_modules.base_bev_backbone import BaseBEVBackbone
from models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from models.sub_modules.downsample_conv import DownsampleConv
from models.sub_modules.naive_compress import NaiveCompressor
from models.fuse_modules.raindrop_attn import raindrop_fuse
from models.sub_modules.where2comm_multisweep import Communication


class PointPillarCoBEVFlow(nn.Module):
    def __init__(self, args, max_cav_num):
        super(PointPillarCoBEVFlow, self).__init__()
        self.B = args['batch_size']
        #self.max_cav = args['max_cav']
        self.max_cav = max_cav_num
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        if 'resnet' in args['base_bev_backbone'] and args['base_bev_backbone']['resnet']:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        if 'num_sweep_frames' in args:    # number of frames we use in LSTM
            self.k = args['num_sweep_frames']
        else:
            self.k = 0

        self.naive_communication = Communication(args['rain_model']['communication'])

        self.single_supervise = False
        self.compensation = False
        self.rain_fusion = raindrop_fuse(args['rain_model'])

        self.multi_scale = args['rain_model']['multi_scale']

        if self.shrink_flag:
            dim = args['shrink_header']['dim'][0]
            self.cls_head = nn.Conv2d(int(dim), args['anchor_number'],
                                    kernel_size=1)
            self.reg_head = nn.Conv2d(int(dim), 7 * args['anchor_number'],
                                    kernel_size=1)
            self.fused_cls_head = nn.Conv2d(int(dim), args['anchor_number'],
                                    kernel_size=1)
            self.fused_reg_head = nn.Conv2d(int(dim), 7 * args['anchor_number'],
                                    kernel_size=1)
        else:
            self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'],
                                    kernel_size=1)
            self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_number'],
                                    kernel_size=1)

        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(128 * 2, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2�� # 384
            self.fused_dir_head = nn.Conv2d(128 * 2, args['dir_args']['num_bins'] * args['anchor_number'],
                                    kernel_size=1)
        else:
            self.use_dir = False

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

        for p in self.fused_cls_head.parameters():
            p.requires_grad = False
        for p in self.fused_reg_head.parameters():
            p.requires_grad = False
        for p in self.rain_fusion.parameters():
            p.requires_grad = False

    def regroup(self, x, record_len, k=1):
        cum_sum_len = torch.cumsum(record_len*k, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        record_frames = data_dict['past_k_time_interval']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        psm_single = self.cls_head(spatial_features_2d)
        rm_single = self.reg_head(spatial_features_2d)
        if self.use_dir:
            dm_single = self.dir_head(spatial_features_2d)

        _, _, K = pairwise_t_matrix.shape[:3]
        if self.multi_scale:
            with_resnet = True if hasattr(self.backbone, 'resnet') else False
            x = self.backbone.resnet.blocks[0](batch_dict['spatial_features']) if with_resnet else self.backbone.blocks[0](batch_dict['spatial_features'])

            batch_confidence_maps = self.regroup(psm_single, record_len, K)
            batch_time_intervals = self.regroup(record_frames, record_len, K)
            _, communication_masks_list, _ = self.naive_communication(batch_confidence_maps, record_len, self.B)
            communication_masks_tensor = torch.concat(communication_masks_list, dim=0)
            if x.shape[-1] != communication_masks_tensor.shape[-1]:
                communication_masks_tensor = F.interpolate(communication_masks_tensor, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
            x = x * communication_masks_tensor

            fused_feature = self.rain_fusion(x, psm_single, record_len, pairwise_t_matrix, record_frames,
                                                             batch_time_intervals, communication_masks_tensor, self.backbone,
                                                             [self.shrink_conv, self.cls_head, self.reg_head], noise_pairwise_t_matrix=None)

            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            batch_node_features = self.regroup(spatial_features_2d, record_len, K)
            batch_confidence_maps = self.regroup(psm_single, record_len, K)
            _, communication_masks, _ = self.naive_communication(batch_confidence_maps, record_len, self.B)

            fused_feature = self.rain_fusion(batch_node_features, psm_single, record_len,
                                                                               pairwise_t_matrix, record_frames, communication_masks_tensor=communication_masks)

        psm = self.fused_cls_head(fused_feature)
        rm = self.fused_reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm}

        if self.use_dir:
            dm = self.fused_dir_head(fused_feature)
            output_dict.update({'dm': dm})

        return output_dict
