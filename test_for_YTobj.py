import os
import numpy as np
import cv2
import copy

import torch
import torch.nn.functional as F
from dataloader.data_for_video import get_testloader
from model.model_for_video import UNet16

import config as config
from logger import *
import metrics

os.environ["CUDA_VISIBLE_DEVICES"] = config.TRAIN['GPU']

folder = os.path.join(config.DATA['data_root'], config.DATA['YTobj_test'])
valid_list = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
res_buffer = {item:[] for item in valid_list}
gt_buffer = {item:[] for item in valid_list}

class_J_buffer = {item:[] for item in valid_list}
class_F_buffer = {item:[] for item in valid_list}

def visual(device, work_dir):
    model = UNet16()
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(config.DATA['best_model'])
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loader = get_testloader(config.DATA['YTobj_test'])

    model.eval()
    PTTM = pttm()

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            PTTM.print_status(0, idx, test_loader)
            image, gt, flow, info, img_for_post = batch
            B = image.shape[0]

            ori_H = info[0][0]
            ori_W = info[0][1]

            image = image.to(device)
            flow = flow.to(device)

            preds = model(image, flow)

            flow = flow.permute(0, 2, 3, 1).cpu().detach()

            res = preds

            for b in range(B):
                res_slice = res[b, :, :, :].unsqueeze(0)
                gt_slice = gt[b, :, :, :].squeeze(0)
                flow_slice = flow[b, :, :, :].squeeze(0)
                ori_image_slice = img_for_post[b, :, :, :].squeeze(0)
                info_slice = info[1][b]

                copy_res_slice = res_slice.clone()
                copy_gt_slice = gt_slice.clone()

                copy_res_slice[copy_res_slice > 0.5] = 1
                copy_res_slice[copy_res_slice <= 0.5] = 0
                copy_gt_slice[copy_gt_slice > 0.5] = 1
                copy_gt_slice[copy_gt_slice <= 0.5] = 0

                res_buffer[info_slice].append(copy_res_slice.cpu().detach())
                gt_buffer[info_slice].append(copy_gt_slice.unsqueeze(0).unsqueeze(0).cpu().detach())
                
                gt_slice = np.asarray(gt_slice, np.float32)
                gt_slice /= (gt_slice.max() + 1e-8)
                
                res_slice = F.upsample(res_slice, size=(ori_H[b].item(), ori_W[b].item()), mode='bilinear', align_corners=False)
                res_slice = res_slice.permute(0, 2, 3, 1).cpu().detach().squeeze(0).squeeze(-1).numpy()
                res_slice = (res_slice - res_slice.min()) / (res_slice.max() - res_slice.min() + 1e-8)

                cat_res = cv2.cvtColor(np.array(res_slice * 255), cv2.COLOR_GRAY2BGR)
                cat_res = cv2.resize(cat_res, dsize=(ori_H[b].item(), ori_W[b].item()), interpolation=cv2.INTER_AREA)
                cat_res = cat_res.astype(np.uint8)
                
                cat_gt = cv2.cvtColor(np.array(gt_slice * 255), cv2.COLOR_GRAY2BGR)
                cat_gt = cv2.resize(cat_gt, dsize=(ori_H[b].item(), ori_W[b].item()), interpolation=cv2.INTER_AREA)
                cat_gt = cat_gt.astype(np.uint8)

                cat_flow = cv2.cvtColor(np.array(flow_slice * 255), cv2.COLOR_RGB2BGR)
                cat_flow = cv2.resize(cat_flow, dsize=(ori_H[b].item(), ori_W[b].item()), interpolation=cv2.INTER_AREA)
                cat_flow = cat_flow.astype(np.uint8)

                cat_ori = cv2.cvtColor(np.array(ori_image_slice), cv2.COLOR_RGB2BGR)
                cat_ori = cv2.resize(cat_ori, dsize=(ori_H[b].item(), ori_W[b].item()), interpolation=cv2.INTER_AREA)
                cat_ori = cat_ori.astype(np.uint8)

                result = cv2.hconcat([cat_ori, cat_flow, cat_res, cat_gt])

                valid_name = info[1][b]
                name = info[2][b]

                total_dir = os.path.join(work_dir, "result", "total", valid_name)
                if not os.path.exists(total_dir):
                    os.makedirs(total_dir)
                
                pred_dir = os.path.join(work_dir, "result", "pred", valid_name)
                if not os.path.exists(pred_dir):
                    os.makedirs(pred_dir)
                
                gt_dir = os.path.join(work_dir, "result", "gt", valid_name)
                if not os.path.exists(gt_dir):
                    os.makedirs(gt_dir)

                cv2.imwrite(os.path.join(total_dir, name), result)
                cv2.imwrite(os.path.join(pred_dir, name), cat_res)
                cv2.imwrite(os.path.join(gt_dir, name), cat_gt)

        print("")

        for name in valid_list:
            metrics_res_J = []
            metrics_res_F = []

            res_masks = torch.cat(res_buffer[name], dim=1)
            gt_masks = torch.cat(gt_buffer[name], dim=1)
            _, L, H, W = res_masks.shape
            all_res_masks = np.zeros((1, L, H, W))
            all_gt_masks = np.zeros((1, L, H, W))

            res_masks_k = copy.deepcopy(res_masks).cpu().numpy()
            res_masks_k[res_masks_k != 1] = 0
            res_masks_k[res_masks_k != 0] = 1
            all_res_masks[0] = res_masks_k[0]

            gt_masks_k = copy.deepcopy(gt_masks).cpu().numpy()
            gt_masks_k[gt_masks_k != 1] = 0
            gt_masks_k[gt_masks_k != 0] = 1
            all_gt_masks[0] = gt_masks_k[0]

            j_metrics_res = np.zeros(all_gt_masks.shape[:2])
            f_metrics_res = np.zeros(all_gt_masks.shape[:2])

            for ii in range(all_gt_masks.shape[0]):
                j_metrics_res[ii] = metrics.db_eval_iou(all_gt_masks[ii], all_res_masks[ii])
                f_metrics_res[ii] = metrics.db_eval_boundary(all_gt_masks[ii], all_res_masks[ii])
                [JM, _, _] = metrics.db_statistics(j_metrics_res[ii])
                metrics_res_J.append(JM)
                [FM, _, _] = metrics.db_statistics(f_metrics_res[ii])
                metrics_res_F.append(FM)
            
            class_J_buffer[name] = np.mean(metrics_res_J)
            class_F_buffer[name] = np.mean(metrics_res_F)
        
        total_J_mean = np.array(list(class_J_buffer.values())).mean()
        total_F_mean = np.array(list(class_F_buffer.values())).mean()
        
        print("Test result: ")
        print("J: {:.4f}, F: {:.4f}, J&F: {:.4f}".format(total_J_mean, total_F_mean, (total_J_mean + total_F_mean) / 2))


def main():
    work_dir = make_new_work_space()
    save_config_file(work_dir)

    print("Check device...")
    device = torch.device("cuda")
    print(device)
    print("ok!")

    visual(device, work_dir)

if __name__ == '__main__':
    main()