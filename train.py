import os
import sys
import argparse
import random
import wandb
import cv2 as cv
import time

from utils import *
from tqdm import tqdm

from dataset import build_dataset
from models.detr import build_model
from loss import build_criterion
from matcher import build_matcher
from lr_sched import adjust_learning_rate

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

#os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '29500'
#os.environ['RANK'] = "0"
#os.environ['WORLD_SIZE'] = "1"

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs.",
        default=None,
        nargs='+',
    )

    # * Optimizer
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # * Train
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--start_eval', default=100, type=int)

    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--num_classes', type=int, default=4,
                        help="Number of cell categories")

    # * Loss
    parser.add_argument('--reg_loss_coef', default=2e-3, type=float)
    parser.add_argument('--cls_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=1.0, type=float,
                        help="Relative classification weight of the no-object class")

    # * Matcher
    parser.add_argument('--set_cost_point', default=0.1, type=float,
                        help="L2 point coefficient in the matching cost")
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")

    # * Model
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--row', default=2, type=int, help="number of anchor points per row")
    parser.add_argument('--col', default=2, type=int, help="number of anchor points per column")

    # * Dataset
    parser.add_argument('--dataset', default='bxr', type=str)
    parser.add_argument('--num_workers', default=0, type=int)

    # * Evaluator
    parser.add_argument('--match_dis', default=12, type=int)

    # * Distributed training
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def construct_dataset():
    dataset_train = build_dataset(args, 'train')
    dataset_val = build_dataset(args, 'test')

    if args.distributed:
        train_sampler = DistributedSampler(dataset_train)
        data_loader_train = DataLoader(dataset_train, sampler=train_sampler, batch_size=args.batch_size,
                                       num_workers=0, collate_fn=collate_fn_pad)
    else:
        data_loader_train = DataLoader(dataset_train, shuffle=True, batch_size=args.batch_size,
                                       num_workers=0, collate_fn=collate_fn_pad)
    data_loader_val = DataLoader(dataset_val, shuffle=False, batch_size=1, num_workers=0,
                                 collate_fn=collate_fn_pad)

    data_loaders = {'train': data_loader_train, 'test': data_loader_val}
    return data_loaders


def train():
    if args.distributed:
        rank = args.gpu
    else:
        rank = 0

    if rank == 0:
        run = wandb.init(project='bxr_cell_recognition', entity="suica46")
        run.name = run.id
        run.save()
    
        cfg = wandb.config
        for k, v in args.__dict__.items():
            setattr(cfg, k, v)

    model = build_model(args).cuda(rank)
    model_without_ddp = model

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)
        model_without_ddp = model.module

    matcher = build_matcher(args)
    criterion = build_criterion(rank, matcher, args)
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    data_loaders = construct_dataset()
    evaluator = Evaluator(data_loaders['test'])
    first_eval = True

    # load checkpoint
    metrics = load_checkpoint(args, model_without_ddp, optimizer) if args.resume else {}
    max_cls_mf1 = metrics.get('分类F1', 0)

    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, data_loaders['train'], optimizer, args.clip_max_norm, epoch, criterion, rank)

        if trigger_eval(epoch, args.start_eval):
            if first_eval:
                save_model(epoch, args, metrics, model_without_ddp, optimizer,mode='best')
                first_eval = False

            metrics = evaluator.calculate_metrics(model, rank=rank, effective_matching_dis=args.match_dis)
            cls_mf1 = metrics['分类指标'][-1]
            print(metrics)
            
            # save most recent epoch model
            save_model(epoch,args,metrics,model_without_ddp,optimizer,mode='recent')
            
            # cls_mf1 = sum(metrics['分类F1'][:2]) / 2
            # print(metrics, cls_mf1)

            if rank == 0:
                wandb.log(dict(zip(["检测精度", "检测召回", "检测F1"], metrics['检测指标'])))
                wandb.log(dict(zip(["分类精度", "分类召回", "分类F1"], metrics['分类指标'])))

                if max_cls_mf1 < cls_mf1:
                    max_cls_mf1 = cls_mf1
                    if args.output_dir:
                        save_model(epoch, args, metrics, model_without_ddp, optimizer,mode='best')

    if args.distributed:
        cleanup()

def test():
    model = build_model(args).cuda()
    model.load_state_dict(torch.load('./checkpoint/_savePath_.pth')['model'])
    model.eval()
    data_loaders = construct_dataset()
    evaluator = Evaluator(data_loaders['test'])
    metrics = evaluator.calculate_metrics(model, rank=0, effective_matching_dis=args.match_dis)
    print(metrics)

def train_one_epoch(model, train_loader, optimizer, max_norm, epoch, criterion, rank):
    model.train()
    if args.distributed:
        train_loader.sampler.set_epoch(epoch)

    iterator = train_loader
    if rank == 0:
        time_string = time.strftime('[%D-%H:%M:%S]',time.localtime())
        iterator = tqdm(train_loader, file=sys.stdout)
        iterator.set_description(f"{time_string} Train epoch-{epoch}")

    reg_tl = cls_tl = 0
    for data_iter_step, (images, points, labels, lengths) in enumerate(iterator):
        # warmup lr
        adjust_learning_rate(args,
                             optimizer,
                             data_iter_step / len(iterator) + epoch,
                             rank)

        images = images.cuda(rank)
        points = points.cuda(rank)
        labels = labels.cuda(rank)

        # all points N×2
        targets = {'gt_nums': lengths,
                   'gt_points': [points_seq[points_seq != -1].reshape(-1, 2) for points_seq in points],
                   'gt_labels': [label_seq[label_seq != -1] for label_seq in labels]}

        cell_ratios = [[(targets['gt_labels'][i] == c).sum().item() for c in range(4)] for i in range(images.size(0))]
        cell_ratios = torch.tensor(cell_ratios, dtype=torch.float).cuda(rank)
        cell_ratios = cell_ratios / cell_ratios.sum(1).unsqueeze(-1)
        targets['cell_ratios'] = cell_ratios

        outputs = model(images)
        losses = criterion(outputs, targets)
        loss = losses.sum()
        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:  # clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        losses /= args.world_size

        if args.distributed:
            dist.all_reduce(losses, op=dist.ReduceOp.SUM)
        reg_tl += losses[0].item()
        cls_tl += losses[1].item()

    print(f'回归损失: {reg_tl}, 分类损失: {cls_tl}')
    if rank == 0:
        wandb.log({
            'reg_loss': reg_tl,
            'cls_loss': cls_tl,
            'tot_loss': reg_tl + cls_tl,
        })


class Evaluator:
    def __init__(self, data_loader_val):
        super(Evaluator, self).__init__()
        self.data_loader = data_loader_val
        self.gds = []
        for sample in data_loader_val.dataset.data:
            self.gds.append(tuple(sample.values())[1:])
        self.num_classes = args.num_classes

    @torch.no_grad()
    def predict(self, model, images, apply_deduplication: bool = False):
        h, w = images.shape[-2:]
        outputs = model(images)

        points = outputs['pnt_coords'][0].cpu().numpy()
        scores = torch.softmax(outputs['cls_logits'][0], dim=-1).cpu().numpy()

        cross_border_index = (points[:, 0] < 0) | (points[:, 0] >= w) | (points[:, 1] < 0) | (points[:, 1] >= h)
        points = points[~cross_border_index]
        scores = scores[~cross_border_index]

        classes = np.argmax(scores, axis=-1)
        reserved_index = classes < args.num_classes

        if apply_deduplication:
            return deduplicate(points[reserved_index], scores[reserved_index], 12)
        else:
            return points[reserved_index], classes[reserved_index]

    def calculate_metrics(self, model, effective_matching_dis=12, rank=0):
        eps = 1e-8
        model.eval()

        cls_pn, cls_tn, cls_rn = list(torch.zeros(self.num_classes).cuda(rank) for _ in range(3))
        det_rn, det_tn, det_pn = list(torch.zeros(1).cuda(rank) for _ in range(3))
        for i, (images, points, labels, lengths) in enumerate(self.data_loader):

            if i % args.world_size != rank:
                continue

            images = images.cuda(non_blocking=True)
            pd_points, pd_classes = self.predict(model, images, apply_deduplication=True)
            gd_points = np.zeros((0, 2), dtype=int)
            for c in range(self.num_classes):
                category_pd_points = pd_points[pd_classes == c]
                category_gd_points = self.gds[i][c]
                #print(pd_points)
                #print(category_pd_points,category_gd_points)
                print(gd_points.shape,category_gd_points.shape)
                gd_points = np.concatenate([gd_points, category_gd_points], axis=0)

                pred_num, gd_num = len(category_pd_points), len(category_gd_points)

                cls_pn[c] += pred_num
                cls_tn[c] += gd_num

                if pred_num and gd_num:
                    right_num, _ = binary_match(category_pd_points, category_gd_points,
                                                threshold_distance=effective_matching_dis)
                    print('right num 1: {}'.format(right_num))
                    cls_rn[c] += right_num

            det_pn += len(pd_points)
            det_tn += len(gd_points)

            if len(pd_points) and len(gd_points):
                right_num, _ = binary_match(pd_points, gd_points, threshold_distance=effective_matching_dis)
                print('right num 2: {}'.format(right_num))
                det_rn += right_num

        if args.distributed:
            dist.all_reduce(det_rn, op=dist.ReduceOp.SUM)
            dist.all_reduce(det_tn, op=dist.ReduceOp.SUM)
            dist.all_reduce(det_pn, op=dist.ReduceOp.SUM)

            dist.all_reduce(cls_pn, op=dist.ReduceOp.SUM)
            dist.all_reduce(cls_tn, op=dist.ReduceOp.SUM)
            dist.all_reduce(cls_rn, op=dist.ReduceOp.SUM)

        det_r = det_rn / (det_tn + eps)
        det_p = det_rn / (det_pn + eps)
        det_f1 = (2 * det_r * det_p) / (det_p + det_r + eps)

        cls_r = cls_rn / (cls_tn + eps)
        cls_p = cls_rn / (cls_pn + eps)
        cls_f1 = (2 * cls_r * cls_p) / (cls_r + cls_p + eps)

        metrics = {'检测指标': [det_p.item(), det_r.item(), det_f1.item()],
                   '分类精度': cls_p.tolist(), '分类召回': cls_r.tolist(), '分类F1': cls_f1.tolist(),
                   '分类指标': [cls_p.mean().item(), cls_r.mean().item(), cls_f1.mean().item()]}
        return metrics

    def visual_analysis(self, model, output_dir='vis_results/normal'):
        from skimage import io

        model.eval()
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]  # PDL1
        # colors = [(102, 255, 0), (255, 0, 0), (234, 255, 0), (0, 238, 255)]  # BXR
        # colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255), (255, 204, 153), (255, 153, 255), # PDL1_all
        #           (204, 0, 255), (153, 255, 204), (0, 255, 255), (204, 153, 0)]

        for i, (images, points, labels, lengths) in enumerate(self.data_loader):
            img_name = self.data_loader.dataset.files[i].split('.')[0]
            print(f'Visualizing --- {img_name}')

            images = images.cuda(non_blocking=True)

            pd_points, pd_classes = self.predict(model, images, apply_deduplication=True)

            # draw pred points
            image = self.data_loader.dataset.data[i]['image'].copy()
            image1 = image.copy()
            for c, (x, y) in zip(pd_classes.astype(int), pd_points.astype(int)):
                cv.line(image, (x - 6, y), (x + 6, y), color=colors[c], thickness=4, lineType=cv.LINE_AA)
                cv.line(image, (x, y - 6), (x, y + 6), color=colors[c], thickness=4, lineType=cv.LINE_AA)
            # io.imsave(f"{output_dir}/{img_name}_pred.png", image, check_contrast=False)

            # draw gd points
            for c in range(self.num_classes):
                for (x, y) in self.gds[i][c].astype(int):
                    cv.circle(image, (x, y), radius=12, color=colors[c], thickness=1, lineType=cv.LINE_AA)
                    # cv.line(image1, (x - 6, y), (x + 6, y), color=colors[c], thickness=4, lineType=cv.LINE_AA)
                    # cv.line(image1, (x, y - 6), (x, y + 6), color=colors[c], thickness=4, lineType=cv.LINE_AA)
            # io.imsave(f"{output_dir}/{img_name}_gd.png", image1, check_contrast=False)

            io.imsave(f"{output_dir}/{img_name}.png", image, check_contrast=False)

    @torch.no_grad()
    def match_procedure(self, model):
        model.eval()
        matcher = build_matcher(args)
        for i, (images, points, labels, lengths) in enumerate(self.data_loader):
            img_name = self.data_loader.dataset.files[i].split('.')[0]
            print(f'Processing --- ({img_name})')

            images = images.cuda(non_blocking=True)
            points = points.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = model(images)

            targets = {'gt_nums': lengths,
                       'gt_points': [points_seq[points_seq != -1].reshape(-1, 2) for points_seq in points],
                       'gt_labels': [label_seq[label_seq != -1] for label_seq in labels]}

            indices = matcher(outputs, targets)
            np.save(f'./tool/result/{img_name}_indices',
                    [indices[0][0].cpu().numpy(), indices[0][1].cpu().numpy()])

            cls_scores = outputs['cls_logits'][0].softmax(-1).cpu().numpy()
            points = outputs['pnt_coords'][0].cpu().numpy()

            reg_attn = outputs['reg_attn'][0, ..., 0].cpu().numpy()
            cls_attn = outputs['cls_attn'][0, ..., 0].cpu().numpy()

            np.save(f'./tool/result/{img_name}_scores', cls_scores)
            np.save(f'./tool/result/{img_name}_points', points.astype(int))
            np.save(f'./tool/result/{img_name}_cls_attn', reg_attn)
            np.save(f'./tool/result/{img_name}_reg_attn', cls_attn)


def eval(ckpt_path):
    model = build_model(args)
    ckpt = torch.load(f'checkpoint/{ckpt_path}', map_location='cpu')
    print(ckpt['epoch'], ckpt['metrics'])
    # ckpt = torch.load(f'./{ckpt_path}', map_location='cpu')

    dataset_val = build_dataset(args, 'test')
    # dataset_val = build_dataset(args, 'train')
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                 collate_fn=collate_fn_pad)

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in ckpt['model'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.cuda()

    evaluator = Evaluator(data_loader_val)
    metrics = evaluator.calculate_metrics(model, effective_matching_dis=args.match_dis)
    print(metrics)

    # evaluator.visual_analysis(model, output_dir='vis_results/ours1')
    # evaluator.match_procedure(model)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    #os.environ['MASTER_PORT'] = '29600'
    
    init_distributed_mode(args)
    

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    train()
    #test()
