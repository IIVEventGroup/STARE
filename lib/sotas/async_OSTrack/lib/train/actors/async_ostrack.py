from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate


class AsyncOSTrackActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):

        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1 + self.cfg.DATA.ASYNC_HORIZON

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        # TODO
        search_img = data['search_images']  # (Horizon, batch, C, H, W)
        # search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond( # work for only 1 template per bench
                self.cfg,
                template_list[0].shape[0],
                template_list[0].device,
                data['template_anno'][0]
            )
            box_mask_z = box_mask_z.repeat(1 + self.cfg.DATA.ASYNC_HORIZON, 1)  # (batch*(1+horizon), *)

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(
                data['epoch'],
                warmup_epochs=ce_start_epoch,
                total_epochs=ce_start_epoch + ce_warm_epoch,
                ITERS_PER_EPOCH=1,
                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0]
            )

        # TODO
        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(
            template=template_list,
            search=search_img,
            ce_template_mask=box_mask_z,
            ce_keep_rate=ce_keep_rate,
            return_last_attn=False
        )

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        L, B = gt_dict['search_anno'].shape[0], gt_dict['search_anno'].shape[1]
        gt_bbox = gt_dict['search_anno'] # (L, B, 4)
        gt_bbox = torch.cat((gt_bbox, gt_bbox[1:]), dim=0).view(-1, 4)  # (L, B, 4) -> ((2L-1)*B, 4)

        gt_gaussian_maps = generate_heatmap(
            gt_dict['search_anno'],
            self.cfg.DATA.SEARCH.SIZE,
            self.cfg.MODEL.BACKBONE.STRIDE
        )
        gt_gaussian_maps = torch.cat(gt_gaussian_maps, dim=0)
        gt_gaussian_maps = torch.cat((gt_gaussian_maps, gt_gaussian_maps[B:]), dim=0).unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")

        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B, N, 4) --> (BN, 4) (x1, y1, x2, y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)

        # compute giou and iou
        try:
            # giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            tea_giou_loss, tea_iou = self.objective['giou'](pred_boxes_vec[:B * L], gt_boxes_vec[:B * L])  # (BN,4) (BN,4)
            stu_giou_loss, stu_iou = self.objective['giou'](pred_boxes_vec[B * L:], gt_boxes_vec[B * L:])  # (BN,4) (BN,4)
        except:
            # giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            tea_giou_loss, tea_iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            stu_giou_loss, stu_iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

        # compute l1 loss
        # l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN, 4) (BN, 4)
        tea_l1_loss = self.objective['l1'](pred_boxes_vec[:B * L], gt_boxes_vec[:B * L])  # (BN, 4) (BN, 4)
        stu_l1_loss = self.objective['l1'](pred_boxes_vec[B * L:], gt_boxes_vec[B * L:])  # (BN, 4) (BN, 4)

        # compute location loss
        if 'score_map' in pred_dict:
            # location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
            tea_location_loss = self.objective['focal'](pred_dict['score_map'][:B * L], gt_gaussian_maps[:B * L])
            stu_location_loss = self.objective['focal'](pred_dict['score_map'][B * L:], gt_gaussian_maps[B * L:])
        else:
            # location_loss = torch.tensor(0.0, device=l1_loss.device)
            tea_location_loss = torch.tensor(0.0, device=tea_l1_loss.device)
            stu_location_loss = torch.tensor(0.0, device=tea_l1_loss.device)

        # weighted sum
        loss = self.loss_weight['giou'] * stu_giou_loss + self.loss_weight['l1'] * stu_l1_loss + self.loss_weight['focal'] * stu_location_loss

        # async block loss of feat distillation
        teacher_feat = pred_dict['backbone_feat'][B:]
        new_feat = pred_dict['async_block_feat']

        # loss_feat_hard = self.objective['feat_hard'](new_feat, teacher_feat.detach())
        loss_feat_hard = self.objective['feat_hard'](new_feat, teacher_feat)
        # loss_feat_soft = 1 - self.objective['feat_soft'](new_feat, teacher_feat.detach(), dim=-1).mean()
        loss_feat_soft = 1 - self.objective['feat_soft'](new_feat, teacher_feat, dim=-1).mean()
        loss += self.loss_weight['feat_hard'] * loss_feat_hard + self.loss_weight['feat_soft'] * loss_feat_soft

        if return_status:
            # status for log
            stu_mean_iou = stu_iou.detach().mean()
            tea_mean_iou = tea_iou.mean()
            status = {
                "Loss/stu_total": loss.item(),
                "Loss/stu_giou": stu_giou_loss.item(),
                "Loss/tea_giou": tea_giou_loss.item(),
                "Loss/stu_l1": stu_l1_loss.item(),
                "Loss/tea_l1": tea_l1_loss.item(),
                "Loss/stu_location": stu_location_loss.item(),
                "Loss/tea_location": tea_location_loss.item(),
                "Loss/feat_hard": loss_feat_hard,
                "Loss/feat_soft": loss_feat_soft,
                "Stu. IoU": stu_mean_iou.item(),
                "Tea. IoU": tea_mean_iou.item(),
            }

            return loss, status

        else:
            return loss
