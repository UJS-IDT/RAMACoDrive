# -*- coding: utf-8 -*-
# Author: Shunyao Zhang <ca19p@163.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn

from models.sub_modules.pillar_vfe import PillarVFE
from models.sub_modules.point_pillar_scatter import PointPillarScatter
from models.sub_modules.base_bev_backbone import BaseBEVBackbone
from models.sub_modules.downsample_conv import DownsampleConv
from models.sub_modules.naive_compress import NaiveCompressor_encoder, NaiveCompressor_decoder
from models.sub_modules.self_attn import AttFusion


class PointPillarOPV2VIntermediate(nn.Module):
    def __init__(self, args, max_cav_num):
        super(PointPillarOPV2VIntermediate, self).__init__()

        #self.max_cav = args['max_cav']
        self.max_cav = max_cav_num
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor_encoder = NaiveCompressor_encoder(256, args['compression'])
            self.naive_compressor_decoder = NaiveCompressor_decoder(256, args['compression'])

        self.fusion_net = AttFusion(256)

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)

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

    def forward(self, data_dict, calculate_flag=False):
        if calculate_flag:
            voxel_features = data_dict['processed_lidar']['voxel_features']
            voxel_coords = data_dict['processed_lidar']['voxel_coords']
            voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
            record_len = data_dict['record_len']

            batch_dict = {'voxel_features': voxel_features,
                          'voxel_coords': voxel_coords,
                          'voxel_num_points': voxel_num_points,
                          'record_len': record_len}
        else:
            batch_dict = data_dict
        
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
            spatial_features_2d = self.naive_compressor_encoder(spatial_features_2d)
        
        if not calculate_flag:
            return spatial_features_2d
        
        # self compressor_decoder
        if self.compression:
            spatial_features_2d = self.naive_compressor_decoder(spatial_features_2d)
        
        if "intermediate_feature_v2x" in data_dict:
            if self.compression:
                for i in range(data_dict["intermediate_feature_v2x"].shape[0]):
                    vehicle_feature = data_dict["intermediate_feature_v2x"][i].unsqueeze(0)
                    decoded_feature = self.naive_compressor_decoder(vehicle_feature)
                    spatial_features_2d = torch.cat((spatial_features_2d, decoded_feature), dim=0)
            else:
                spatial_features_2d = torch.cat((spatial_features_2d, data_dict["intermediate_feature_v2x"]), dim=0)
        
        fused_feature = self.fusion_net(spatial_features_2d, record_len)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict
