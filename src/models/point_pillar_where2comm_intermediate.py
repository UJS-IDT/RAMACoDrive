# -*- coding: utf-8 -*-
# Author: Shunyao Zhang <ca19p@163.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sub_modules.pillar_vfe import PillarVFE
from models.sub_modules.point_pillar_scatter import PointPillarScatter
from models.sub_modules.base_bev_backbone import BaseBEVBackbone
from models.sub_modules.downsample_conv import DownsampleConv
from models.sub_modules.naive_compress import NaiveCompressor_encoder, NaiveCompressor_decoder
from models.fuse_modules.where2comm_fuse_full import Where2comm, Communication


class PointPillarWhere2commIntermediate(nn.Module):
    def __init__(self, args, max_cav_num):
        super(PointPillarWhere2commIntermediate, self).__init__()

        #self.max_cav = args['max_cav']
        self.max_cav = max_cav_num
        self.B = args['batch_size']
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

        if 'training_flag' in args:
            args['where2comm_fusion']['communication']['training_flag'] = args['training_flag']

        self.fusion_net = Where2comm(args['where2comm_fusion'])
        self.multi_scale = args['where2comm_fusion']['multi_scale']
        self.fully = args['where2comm_fusion']['fully']
        self.naive_communication = Communication(args['where2comm_fusion']['communication'])

        self.cls_head = nn.Conv2d(args['head_dim'], args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(args['head_dim'], 7 * args['anchor_number'], kernel_size=1)

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

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def restore_tensor(self, valid_elements, valid_indices, original_shape):
        restored_tensor = torch.zeros(original_shape, dtype=valid_elements.dtype).cuda()
        restored_tensor.view(-1)[valid_indices] = valid_elements
        return restored_tensor

    def forward(self, data_dict, calculate_flag=False):
        if calculate_flag:
            voxel_features = data_dict['processed_lidar']['voxel_features']
            voxel_coords = data_dict['processed_lidar']['voxel_coords']
            voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
            record_len = data_dict['record_len']
            pairwise_t_matrix = data_dict['pairwise_t_matrix']
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

        psm_single = self.cls_head(spatial_features_2d)

        if self.multi_scale:
            # Bypass communication cost, communicate at high resolution, neither shrink nor compress
            x = self.backbone.blocks[0](batch_dict['spatial_features'])
            if not self.fully:
                B = self.B
                batch_confidence_maps = torch.tensor_split(psm_single, [])
                communication_masks, _ = self.naive_communication(batch_confidence_maps, B)
                if x.shape[-1] != communication_masks.shape[-1]:
                    communication_masks = F.interpolate(communication_masks, size=(x.shape[-2], x.shape[-1]),
                                                        mode='bilinear', align_corners=False)

                x = x * communication_masks
                mask = (x != 0).to(torch.uint8)
                x_trans = x[mask.bool()]
                flat_mask = mask.view(-1)
                valid_indices = flat_mask.nonzero(as_tuple=True)[0].to(torch.int32)
            else:
                mask = (x != 0).to(torch.uint8)
                x_trans = x[mask.bool()]
                flat_mask = mask.view(-1)
                valid_indices = flat_mask.nonzero(as_tuple=True)[0].to(torch.int32)
                
            if not calculate_flag:
                return x_trans, valid_indices

            if "intermediate_feature_v2x" in data_dict:
                for i in range(len(data_dict["intermediate_feature_v2x"])):
                    restored_x = self.restore_tensor(data_dict["intermediate_feature_v2x"][i], data_dict["feature_mask_v2x"][i], mask.shape)
                    x = torch.cat((x, restored_x), dim=0)

            fused_feature = self.fusion_net(x, psm_single, record_len, pairwise_t_matrix, self.backbone)

            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)

        else:
            # compressor
            if self.compression:
                spatial_features_2d = self.naive_compressor_encoder(spatial_features_2d)

            if not self.fully:
                B = self.B
                batch_confidence_maps = torch.tensor_split(psm_single, [])
                communication_masks, _ = self.naive_communication(batch_confidence_maps, B)
                x = spatial_features_2d * communication_masks
            else:
                x = spatial_features_2d

            if not calculate_flag:
                return x

            # self compressor_decoder
            if self.compression:
                x = self.naive_compressor_decoder(x)

            if "intermediate_feature_v2x" in data_dict:
                if self.compression:
                    for i in range(data_dict["intermediate_feature_v2x"].shape[0]):
                        vehicle_feature = data_dict["intermediate_feature_v2x"][i].unsqueeze(0)
                        decoded_feature = self.naive_compressor_decoder(vehicle_feature)
                        x = torch.cat((x, decoded_feature), dim=0)
                else:
                    x = torch.cat((x, data_dict["intermediate_feature_v2x"]), dim=0)

            fused_feature = self.fusion_net(x, psm_single, record_len, pairwise_t_matrix)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict
