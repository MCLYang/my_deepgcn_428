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

from opt import OptInit
from architecture import DenseDeepGCN
from utils.ckpt_util import load_pretrained_models, load_pretrained_optimizer, save_checkpoint
from utils.metrics import AverageMeter
from utils import optim
import pdb
import numpy as np
from BigredDataSet import BigredDataset
from tqdm import tqdm
import time

def get_weight(label_set):
    #input is the target tensor
    #output is 1*N array
    labelweights, _ = np.histogram(label_set, range(3))
    labelweights = labelweights.astype(np.float32)
    labelweights = labelweights / np.sum(labelweights)
    labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
    return(labelweights)
#

def main():
    NUM_POINT = 20000
    opt = OptInit().initialize()
    opt.num_worker = 32
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpuNum

    opt.printer.info('===> Creating dataloader ...')

    train_dataset = BigredDataset(root = opt.train_path,
                                 is_train=True,
                                 is_validation=False,
                                 is_test=False,
                                 num_channel=opt.num_channel,
                                 pre_transform=T.NormalizeScale()
                                 )
    train_loader = DenseDataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_worker)

    validation_dataset = BigredDataset(root = opt.train_path,
                                 is_train=False,
                                 is_validation=True,
                                 is_test=False,
                                 num_channel=opt.num_channel,
                                 pre_transform=T.NormalizeScale()
                                 )
    validation_loader = DenseDataLoader(validation_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_worker)

    opt.printer.info('===> computing Labelweight ...')

    labelweights = np.zeros(2)
    labelweights, _ = np.histogram(train_dataset.data.y.numpy(), range(3))
    labelweights = labelweights.astype(np.float32)
    labelweights = labelweights / np.sum(labelweights)
    labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
    weights = torch.Tensor(labelweights).cuda()
    print("labelweights", weights)

    opt.n_classes = train_loader.dataset.num_classes

    opt.printer.info('===> Loading the network ...')

    opt.best_value = 0
    print("GPU:",opt.device)
    model = DenseDeepGCN(opt).to(opt.device)
    if opt.multi_gpus:
        model = DataParallel(DenseDeepGCN(opt)).to(device=opt.device)
    opt.printer.info('===> loading pre-trained ...')
    # model, opt.best_value, opt.epoch = load_pretrained_models(model, opt.pretrained_model, opt.phase)

    opt.printer.info('===> Init the optimizer ...')
    criterion = torch.nn.CrossEntropyLoss(weight = weights).to(opt.device)
    # criterion_test = torch.nn.CrossEntropyLoss(weight = weights)

    if opt.optim.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optim.lower() == 'radam':
        optimizer = optim.RAdam(model.parameters(), lr=opt.lr)
    else:
        raise NotImplementedError('opt.optim is not supported')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_adjust_freq, opt.lr_decay_rate)
    # optimizer, scheduler, opt.lr = load_pretrained_optimizer(opt.pretrained_model, optimizer, scheduler, opt.lr)

    opt.printer.info('===> Init Metric ...')
    opt.losses = AverageMeter()
    # opt.test_metric = miou
    opt.test_values = AverageMeter()
    opt.test_value = 0.

    opt.printer.info('===> start training ...')
    writer = SummaryWriter()
    writer_test = SummaryWriter()
    counter_test = 0
    counter_play = 0
    start_epoch = 0
    mean_miou = AverageMeter()
    mean_loss =  AverageMeter()
    mean_acc =  AverageMeter()
    best_value = 0
    for epoch in range(start_epoch, opt.total_epochs):
        opt.epoch += 1
        model.train()
        total_seen_class = [0 for _ in range(opt.n_classes)]
        total_correct_class = [0 for _ in range(opt.n_classes)]
        total_iou_deno_class = [0 for _ in range(opt.n_classes)]
        ave_mIoU = 0
        total_correct = 0
        total_seen = 0
        loss_sum = 0

        mean_miou.reset()
        mean_loss.reset()
        mean_acc.reset()


        for i, data in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
            # if i % 50 == 0:
            opt.iter += 1
            if not opt.multi_gpus:
                data = data.to(opt.device)
            target = data.y
            batch_label2 = target.cpu().data.numpy()
            inputs = torch.cat((data.pos.transpose(2, 1).unsqueeze(3), data.x.transpose(2, 1).unsqueeze(3)), 1)
            inputs = inputs[:, :opt.num_channel, :, :]
            gt = data.y.to(opt.device)
            # ------------------ zero, output, loss
            optimizer.zero_grad()
            out = model(inputs)

            loss = criterion(out, gt)
            #pdb.set_trace()

            # ------------------ optimization
            loss.backward()
            optimizer.step()

            seg_pred= out.transpose(2,1)

            pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred.contiguous().view(-1, opt.n_classes)
            #pdb.set_trace()
            pred_val = np.argmax(pred_val, 2)
            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)

            total_correct += correct
            total_seen += (opt.batch_size *NUM_POINT)
            loss_sum += loss.item()

            current_seen_class = [0 for _ in range(opt.n_classes)]
            current_correct_class = [0 for _ in range(opt.n_classes)]
            current_iou_deno_class = [0 for _ in range(opt.n_classes)]
            #pdb.set_trace()

            for l in range(opt.n_classes):
                #pdb.set_trace()
                total_seen_class[l] += np.sum((batch_label2 == l))
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label2 == l))
                total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label2 == l)))
                current_seen_class[l] = np.sum((batch_label2 == l))
                current_correct_class[l] = np.sum((pred_val == l) & (batch_label2 == l))
                current_iou_deno_class[l] = np.sum(((pred_val == l) | (batch_label2 == l)))

            #pdb.set_trace()
            writer.add_scalar('training_loss', loss.item(), counter_play)
            writer.add_scalar('training_accuracy', correct / float(opt.batch_size * NUM_POINT), counter_play)
            m_iou = np.mean(np.array(current_correct_class) / (np.array(current_iou_deno_class, dtype=np.float) + 1e-6))
            writer.add_scalar('training_mIoU', m_iou, counter_play)
            ave_mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))

            # print("training_loss:",loss.item())
            # print('training_accuracy:',correct / float(opt.batch_size * NUM_POINT))
            # print('training_mIoU:',m_iou)

            mean_miou.update(m_iou)
            mean_loss.update(loss.item())
            mean_acc.update(correct / float(opt.batch_size * NUM_POINT))

            counter_play = counter_play + 1

        train_mIoU = mean_miou.avg
        train_macc = mean_acc.avg
        train_mloss = mean_loss.avg

        print('Epoch: %d, Training point avg class IoU: %f' % (epoch,train_mIoU))
        print('Epoch: %d, Training mean loss: %f' %(epoch, train_mloss))
        print('Epoch: %d, Training accuracy: %f' %(epoch, train_macc))

        mean_miou.reset()
        mean_loss.reset()
        mean_acc.reset()

        print('validation_loader')

        model.eval()
        with torch.no_grad():
            for i, data in tqdm(enumerate(validation_loader), total=len(validation_loader), smoothing=0.9):
                # if i % 50 ==0:
                if not opt.multi_gpus:
                    data = data.to(opt.device)

                target = data.y
                batch_label2 = target.cpu().data.numpy()

                inputs = torch.cat((data.pos.transpose(2, 1).unsqueeze(3), data.x.transpose(2, 1).unsqueeze(3)), 1)
                inputs = inputs[:, :opt.num_channel, :, :]
                gt = data.y.to(opt.device)
                out = model(inputs)
                loss = criterion(out, gt)
                #pdb.set_trace()

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

                    current_seen_class[l] = np.sum((batch_label2 == l))
                    current_correct_class[l] = np.sum((pred_val == l) & (batch_label2 == l))
                    current_iou_deno_class[l] = np.sum(((pred_val == l) | (batch_label2 == l)))
                m_iou = np.mean(
                    np.array(current_correct_class) / (np.array(current_iou_deno_class, dtype=np.float) + 1e-6))
                mean_miou.update(m_iou)
                mean_loss.update(loss.item())
                mean_acc.update(correct / float(opt.batch_size * NUM_POINT))

        validation_mIoU = mean_miou.avg
        validation_macc = mean_acc.avg
        validation_mloss = mean_loss.avg
        writer.add_scalar('validation_loss', validation_mloss, epoch)
        print('Epoch: %d, validation mean loss: %f' %(epoch, validation_mloss))
        writer.add_scalar('validation_accuracy', validation_macc, epoch)
        print('Epoch: %d, validation accuracy: %f' %(epoch, validation_macc))
        writer.add_scalar('validation_mIoU', validation_mIoU, epoch)
        print('Epoch: %d, validation point avg class IoU: %f' % (epoch,validation_mIoU))

        model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        package ={
        'epoch': opt.epoch,
        'state_dict': model_cpu,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_miou':train_mIoU,
        'train_accuracy':train_macc,
        'train_loss':train_mloss,
        'validation_mIoU':validation_mIoU,
        'validation_macc':validation_macc,
        'validation_mloss':validation_mloss,
        'num_channel':opt.num_channel,
        'gpuNum': opt.gpuNum,
        'time':time.ctime()
        }
        torch.save(package,'saves/val_miou_%f_val_acc_%f_%d.pth' % (validation_mIoU, validation_macc, epoch))
        is_best = (best_value < validation_mIoU)
        print('Is Best? ',is_best)
        if (best_value < validation_mIoU):
            best_value = validation_mIoU
            torch.save(package,'saves/best_model.pth')
        print('Best IoU: %f' % (best_value))
        scheduler.step()
    opt.printer.info('Saving the final model.Finish!')


# def train(model, train_loader, optimizer, scheduler, criterion, opt):
#     model.train()
#     for i, data in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
#         opt.iter += 1
#         if not opt.multi_gpus:
#             data = data.to(opt.device)
#         inputs = torch.cat((data.pos.transpose(2, 1).unsqueeze(3), data.x.transpose(2, 1).unsqueeze(3)), 1)
#         inputs = inputs[:,:5,:,:]
#         gt = data.y.to(opt.device)
#         # ------------------ zero, output, loss
#         optimizer.zero_grad()
#         out = model(inputs)
#         loss = criterion(out, gt,weights)
#
#         # ------------------ optimization
#         loss.backward()
#         optimizer.step()
#
#         opt.losses.update(loss.item())
#         # ------------------ show information
#         if opt.iter % opt.print_freq == 0:
#             opt.printer.info('Epoch:{}\t Iter: {}\t [{}/{}]\t Loss: {Losses.avg: .4f}'.format(
#                 opt.epoch, opt.iter, i + 1, len(train_loader), Losses=opt.losses))
#             opt.losses.reset()
#
#         # ------------------ tensor board log
#         info = {
#             'loss': loss.item(),
#             #'test_value': opt.test_value,
#             'lr': scheduler.get_lr()[0]
#         }
#         # pdb.set_trace()
#         # print(info)
#         for tag, value in info.items():
#             opt.logger.scalar_summary(tag, value, opt.iter)
#
#         current_seen_class = [0 for _ in range(opt.n_classes)]
#         current_correct_class = [0 for _ in range(opt.n_classes)]
#         current_iou_deno_class = [0 for _ in range(opt.n_classes)]
#         for l in range(opt.n_classes):
#             total_seen_class[l] += np.sum((batch_label2 == l))
#             total_correct_class[l] += np.sum((pred_val == l) & (batch_label2 == l))
#             total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label2 == l)))
#             current_seen_class[l] = np.sum((batch_label2 == l))
#             current_correct_class[l] = np.sum((pred_val == l) & (batch_label2 == l))
#             current_iou_deno_class[l] = np.sum(((pred_val == l) | (batch_label2 == l)))
#
#         writer.add_scalar('training loss', loss.item(), counter_play)
#         writer.add_scalar('training_accuracy', correct / float(BATCH_SIZE * NUM_POINT), counter_play)
#         m_iou = np.mean(np.array(current_correct_class) / (np.array(current_iou_deno_class, dtype=np.float) + 1e-6))
#         writer.add_scalar('training_mIoU', m_iou, counter_play)
#
#         counter_play = counter_play + 1
#
#     # ------------------ save checkpoints
#     # min or max. based on the metrics
#     # is_best = (opt.test_value < opt.best_value)
#     # opt.best_value = min(opt.test_value, opt.best_value)
#
#     model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
#     save_checkpoint({
#         'epoch': opt.epoch,
#         'state_dict': model_cpu,
#         'optimizer_state_dict': optimizer.state_dict(),
#         'scheduler_state_dict': scheduler.state_dict(),
#         'best_value': opt.best_value,
#     }, is_best, opt.save_path, opt.post)
#
#
# def test(model, test_loader, test_metric, opt):
#     opt.test_values.reset()
#     model.eval()
#     with torch.no_grad():
#         for i, data in enumerate(test_loader):
#             if not opt.multi_gpus:
#                 data = data.to(opt.device)
#             inputs = torch.cat((data.pos.transpose(2, 1).unsqueeze(3), data.x.transpose(2, 1).unsqueeze(3)), 1)
#             gt = data.y.to(opt.device)
#
#             out = opt.model(inputs)
#             test_value = test_metric(out.max(dim=1)[1], gt, opt.n_classes)
#             opt.test_values.update(test_value, opt.batch_size)
#         opt.printer.info('Epoch: [{0}]\t Iter: [{1}]\t''TEST loss: {test_values.avg: .4f})\t'.format(
#                opt.epoch, opt.iter, test_values=opt.test_values))
#
#     opt.test_value = opt.test_values.avg


if __name__ == '__main__':
    main()


