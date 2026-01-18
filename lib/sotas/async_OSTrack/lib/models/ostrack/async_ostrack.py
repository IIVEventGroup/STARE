"""
Basic OSTrack model.
"""
import math
import os
import time
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.ostrack.vit import vit_base_patch16_224, async_transformer_block_for_vit_base_patch16_224_ce
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh


class AsyncOSTrack(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, vit, box_head, async_transformer, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            vit: torch module of the vit architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = vit
        self.box_head = box_head
        self.async_block = async_transformer

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        self.patch_embed = None
        self.pos_embed_x = None
        self.pos_drop = None

        self.teacher_patch_embed_z = None
        self.teacher_patch_embed_x = None
        self.teacher_feat = None

    def forward(
            self,
            template: torch.Tensor,
            search: torch.Tensor,
            ce_template_mask=None,
            ce_keep_rate=None,
            return_last_attn=False,
    ):
        with torch.no_grad():
            x, aux_dict = self.backbone(
                z=template,
                x=search,
                ce_template_mask=ce_template_mask,
                ce_keep_rate=ce_keep_rate,
                return_last_attn=return_last_attn,
            )

        # Forward head
        feat_last = x  # [batch_size, 320, 768]
        if isinstance(x, list):
            feat_last = x[-1]

        # get condition info from backbone
        teacher_patch_embed_z = aux_dict.get("teacher_patch_embed_z", None)  # (L * B, *, Dim)
        teacher_patch_embed_x = aux_dict.get("teacher_patch_embed_x", None)  # (L * B, *, Dim)

        # patch_embed = aux_dict.get("patch_embed", None)
        # pos_embed_x = aux_dict.get("pos_embed_x", None)
        # pos_drop = aux_dict.get("pos_drop", None)

        L, B, embed_dim = search.shape[0], search.shape[1], feat_last.shape[-1]
        new_patch_embed_x = teacher_patch_embed_x[B:]
        teacher_patch_embed_z = teacher_patch_embed_z[B:]
        teacher_patch_embed_x = teacher_patch_embed_x[B:]

        teacher_feat = feat_last[B:]
        new_feat = torch.cat((
            teacher_patch_embed_z,
            teacher_patch_embed_x,
            new_patch_embed_x,
            teacher_feat,
        ), dim=1)

        # get prediction from async block
        new_feat = self.async_block(new_feat)
        N = teacher_feat.shape[1]
        new_feat = new_feat[:, -N:]

        all_feat = torch.cat((feat_last, new_feat), dim=0)
        out = self.forward_head(all_feat, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        out['async_block_feat'] = new_feat

        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {
                'pred_boxes': outputs_coord_new,
                'score_map': score_map,
            }

            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {
                'pred_boxes': outputs_coord_new,
                'score_map': score_map_ctr,
                'size_map': size_map,
                'offset_map': offset_map,
            }

            return out

        else:
            raise NotImplementedError

    def forward_slow(
            self,
            template: torch.Tensor,
            search: torch.Tensor,
            ce_template_mask=None,
            ce_keep_rate=None,
            return_last_attn=False,
    ):

        # last_time = time.perf_counter()

        x, aux_dict = self.backbone(
            z=template,
            x=search,
            ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate,
            return_last_attn=return_last_attn,
        )

        # curr_time = time.perf_counter()
        # elapsed_time = curr_time - last_time
        # last_time = curr_time
        # print(f"Foward backbone time:", elapsed_time)

        # Forward head
        feat_last = x  # [batch_size, 320, 768]
        if isinstance(x, list):
            feat_last = x[-1]

        # for async tracking block
        self.teacher_patch_embed_z = aux_dict.get("teacher_patch_embed_z", None)
        self.teacher_patch_embed_x = aux_dict.get("teacher_patch_embed_x", None)
        self.patch_embed = aux_dict.get("patch_embed", None)
        self.pos_embed_x = aux_dict.get("pos_embed_x", None)
        self.pos_drop = aux_dict.get("pos_drop", None)
        self.teacher_feat = feat_last

        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x

        # curr_time = time.perf_counter()
        # elapsed_time = curr_time - last_time
        # last_time = curr_time
        # print(f"Foward head time:", elapsed_time)

        return out

    def forward_fast(
            self,
            new_x: torch.Tensor,
    ):
        new_patch_embed_x = self.patch_embed(new_x)
        new_patch_embed_x += self.pos_embed_x
        new_patch_embed_x = self.pos_drop(new_patch_embed_x)
        new_feat = torch.cat((
            self.teacher_patch_embed_z,
            self.teacher_patch_embed_x,
            new_patch_embed_x,
            self.teacher_feat,
        ), dim=1)

        new_feat = self.async_block(new_feat)
        N = self.teacher_feat.shape[1]
        new_feat = new_feat[:, -N:]

        # TODO: test if updating self.teacher_feat and self.teacher_patch_embed_x will be better
        # self.teacher_feat = new_feat
        # self.teacher_patch_embed_x = new_patch_embed_x

        out = self.forward_head(new_feat, None)
        out['async_block_feat'] = new_feat

        return out

def build_async_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root

    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        raise NotImplementedError

        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(
            pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
        )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

        async_block = async_transformer_block_for_vit_base_patch16_224_ce(
            pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
        )

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        raise NotImplementedError

        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = AsyncOSTrack(
        backbone,
        box_head,
        async_block,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
