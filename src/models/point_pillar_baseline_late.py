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


class PointPillarBaselineLate(nn.Module):
    def __init__(self, args, max_cav_num):
        super(PointPillarBaselineLate, self).__init__()

        # self.max_cav = args['max_cav']
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
        self.score_threshold = args['score_threshold']
        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor_encoder = NaiveCompressor_encoder(256, args['compression'])
            self.naive_compressor_decoder = NaiveCompressor_decoder(256, args['compression'])

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)

        # self.cls_head = nn.Conv2d(self.max_cav, 1, kernel_size=1)
        # self.reg_head = nn.Conv2d(self.max_cav, 1, kernel_size=1)

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
        rm_single = self.reg_head(spatial_features_2d)

        prob = F.sigmoid(psm_single)
        mask = torch.gt(prob, self.score_threshold)
        mask_reg = mask.repeat(1, 7, 1, 1)

        if not calculate_flag:
            psm_selected = torch.masked_select(psm_single, mask)
            rm_selected = torch.masked_select(rm_single, mask_reg)
            scores = torch.masked_select(prob, mask)
            mask_flattened = mask.view(-1)
            mask_reg_flattened = mask_reg.view(-1)
            mask_indices = mask_flattened.nonzero(as_tuple=True)[0].to(torch.int32)
            mask_reg_indices = mask_reg_flattened.nonzero(as_tuple=True)[0].to(torch.int32)

            result_single = {"psm_selected": psm_selected,
                             "rm_selected": rm_selected,
                             "scores": scores,
                             "mask_indices": mask_indices,
                             "mask_reg_indices": mask_reg_indices}
            return result_single

        if "psm_result_v2x" in data_dict:
            for i in range(len(data_dict["psm_result_v2x"])):
                restored_psm = self.restore_tensor(data_dict["psm_result_v2x"][i], data_dict["mask_v2x"][i], mask.shape)
                restored_rm = self.restore_tensor(data_dict["rm_result_v2x"][i], data_dict["mask_reg_v2x"][i],
                                                  mask_reg.shape)
                restored_scores = self.restore_tensor(data_dict["scores_v2x"][i], data_dict["mask_v2x"][i], mask.shape)
                psm_single = torch.cat((psm_single, restored_psm), dim=0)
                rm_single = torch.cat((rm_single, restored_rm), dim=0)
                prob = torch.cat((prob, restored_scores), dim=0)
            values, indices = torch.max(prob, dim=0)
            psm_single_perm = psm_single.permute(1, 2, 3, 0)
            indices_expanded = indices.unsqueeze(-1)
            psm = torch.gather(psm_single_perm, dim=3, index=indices_expanded)
            psm = psm.squeeze(-1)
            psm = psm.unsqueeze(0)
            indices_reg = indices.repeat(7, 1, 1)
            rm_single_perm = rm_single.permute(1, 2, 3, 0)
            indices_reg_expanded = indices_reg.unsqueeze(-1)
            rm = torch.gather(rm_single_perm, dim=3, index=indices_reg_expanded)
            rm = rm.squeeze(-1)
            rm = rm.unsqueeze(0)
            output_dict = {'psm': psm,
                           'rm': rm}
        else:
            output_dict = {'psm': psm_single,
                           'rm': rm_single}

        return output_dict
