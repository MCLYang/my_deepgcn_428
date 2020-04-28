import os
import sys
import torch
import torch_geometric.datasets as GeoData
from torch_geometric.data import DenseDataLoader
from torch.utils.tensorboard import SummaryWriter
import torch_geometric.transforms as T
from torch.nn import DataParallel
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from architecture import DenseDeepGCN
from utils.ckpt_util import load_pretrained_models, load_pretrained_optimizer, save_checkpoint
from utils.metrics import AverageMeter
from utils import optim
import pdb
import numpy as np
from BigredDataSet import BigredDataset
from tqdm import tqdm

from opt_test import OptInit
import time

NUM_POINT = 20000


def main():
    opt = OptInit().initialize()
    opt.batch_size = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpuNum

    print('===> Creating dataloader...')
    # def __init__(self,
    #              root,
    #              is_train=True,
    #              is_validation=False,
    #              is_test=False,
    #              num_channel=5,
    #              pre_transform=None,
    #              pre_filter=None)
    test_dataset = BigredDataset(root = opt.test_path,
                                 is_train=False,
                                 is_validation=False,
                                 is_test=True,
                                 num_channel=5,
                                 pre_transform=T.NormalizeScale()
                                 )
    test_loader = DenseDataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=32)
    opt.n_classes = 2

    print('len(test_loader):',len(test_loader))
    print('phase: ',opt.phase)
    print('batch_size: ',opt.batch_size)
    print('use_cpu: ',opt.use_cpu)
    print('gpuNum: ',opt.gpuNum)
    print('multi_gpus: ',opt.multi_gpus)
    print('test_path: ',opt.test_path)
    print('in_channels: ',opt.in_channels)
    print('device: ',opt.device)
    

    print('===> Loading the network ...')
    model = DenseDeepGCN(opt).to(opt.device)
    load_package = torch.load(opt.pretrained_model)
    model, opt.best_value, opt.epoch = load_pretrained_models(model, opt.pretrained_model, opt.phase)
    pdb.set_trace()
    for item in load_package.keys():
        if(item != 'optimizer_state_dict' and item != 'state_dict' and item != 'scheduler_state_dict'):
            print(str(item),load_package[item])

    print('===> Start Evaluation ...')
    test(model, test_loader, opt)

def test(model, loader, opt):
    mean_miou = AverageMeter()
    mean_loss = AverageMeter()
    mean_acc = AverageMeter()
    mean_time = AverageMeter()
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
            # if i % 50 ==0:
            if not opt.multi_gpus:
                data = data.to(opt.device)
            target = data.y
            batch_label2 = target.cpu().data.numpy()
            inputs = torch.cat((data.pos.transpose(2, 1).unsqueeze(3), data.x.transpose(2, 1).unsqueeze(3)), 1)
            inputs = inputs[:, :self.num_channel, :, :]
            gt = data.y.to(opt.device)
            tic = time.perf_counter()
            out = model(inputs)
            toc = time.perf_counter()
            print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")

            # loss = criterion(out, gt)
            # pdb.set_trace()

            seg_pred = out.transpose(2, 1)
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred.contiguous().view(-1, opt.n_classes)
            pred_val = np.argmax(pred_val, 2)
            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            current_seen_class = [0 for _ in range(opt.n_classes)]
            current_correct_class = [0 for _ in range(opt.n_classes)]
            current_iou_deno_class = [0 for _ in range(opt.n_classes)]
            for l in range(opt.n_classes):
                # pdb.set_trace()
                current_seen_class[l] = np.sum((batch_label2 == l))
                current_correct_class[l] = np.sum((pred_val == l) & (batch_label2 == l))
                current_iou_deno_class[l] = np.sum(((pred_val == l) | (batch_label2 == l)))
            m_iou = np.mean(
                np.array(current_correct_class) / (np.array(current_iou_deno_class, dtype=np.float) + 1e-6))
            mean_time.update(toc - tic)
            mean_miou.update(m_iou)
            mean_acc.update(correct / float(opt.batch_size * NUM_POINT))
            # pdb.set_trace()
    test_time = mean_time.avg
    test_mIoU = mean_miou.avg
    test_macc = mean_acc.avg
    print('Test point avg class IoU: %f' % (test_mIoU))
    print('Test accuracy: %f' % (test_macc))
    print('Test ave time(sec/frame): %f' % (test_time))
    print('Test ave time(frame/sec): %f' % (1/test_time))





    # model.eval()
    # with torch.no_grad():
    #     for i, data in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
    #         target = data.y
    #         inputs = torch.cat((data.pos.transpose(2, 1).unsqueeze(3), data.x.transpose(2, 1).unsqueeze(3)), 1)
    #         gt = data.y
    # #
    #         out = model(inputs)
    #         pred = out.max(dim=1)[1]
    #
    #         pred_np = pred.cpu().numpy()
    #         target_np = gt.cpu().numpy()
    #
    #         for cl in range(opt.n_classes):
    #             cur_gt_mask = (target_np == cl)
    #             cur_pred_mask = (pred_np == cl)
    #             I = np.sum(np.logical_and(cur_pred_mask, cur_gt_mask), dtype=np.float32)
    #             U = np.sum(np.logical_or(cur_pred_mask, cur_gt_mask), dtype=np.float32)
    #             Is[i, cl] = I
    #             Us[i, cl] = U
    #
    # ious = np.divide(np.sum(Is, 0), np.sum(Us, 0))
    # ious[np.isnan(ious)] = 1
    # for cl in range(opt.n_classes):
    #     print("===> mIOU for class {}: {}".format(cl, ious[cl]))
    # print("===> mIOU is {}".format(np.mean(ious)))
    #

if __name__ == '__main__':
    main()


