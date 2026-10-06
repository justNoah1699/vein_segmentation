#!/usr/bin/env python3
"""Fine-tune the pretrained vein model on the labeled test/ images.

modes:
  split : leaf-grouped train/val/final split; best epoch picked on val, final evaluated once
  cv    : leaf-grouped k-fold, fixed epochs, compares pretrained vs fine-tuned per fold
  all   : train on all labeled images for a fixed number of epochs and save the model
"""
import argparse
import string
import json
from datetime import datetime
import copy
import os
import os.path as osp
import random
import re

import glog as log
import numpy as np
import torch
from torch.utils.data import DataLoader

from networks.corenet import CoRE_Net
from networks.framework import MyFrame
from utils.data_selftrain import ImageFolder
from utils.loss import dice_bce_loss
from utils.metrics import calculate_IoU_Dice


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['split', 'cv', 'all'], default='cv')
    p.add_argument('--data-root', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--pretrained', required=True)
    p.add_argument('--out-dir', default='./results/finetune')
    p.add_argument('--epochs', default=30, type=int)
    p.add_argument('--batch-size', default=4, type=int)
    p.add_argument('--learning-rate', default=1e-5, type=float)
    p.add_argument('--folds', default=5, type=int, help='cv only')
    p.add_argument('--train-frac', default=0.6, type=float, help='split only')
    p.add_argument('--val-frac', default=0.2, type=float, help='split only')
    p.add_argument('--seed', default=0, type=int)
    # needed by MyFrame
    p.add_argument('--pos-weight', default=1, type=float)
    p.add_argument('--max-norm', default=1.0, type=float)
    p.add_argument('--lambda-dice-iou-loss', default=0.5, type=float)
    p.add_argument('--num-subdivision-points', default=28 * 28, type=int)
    p.add_argument('--threshold', default=0.9, type=float)
    return p.parse_args()


# leaf id: drop CEE_DE_<site>_P<timestamp>_ prefix and the 'leaf_' token
def key(name):
    return re.sub(r'^CEE_DE_\w+?_P\d+_', '', name).replace('leaf_', '')


def subset(ds, idx, is_random):
    s = copy.copy(ds)
    for a in ['images', 'outlines', 'veins', 'image_names']:
        setattr(s, a, [getattr(ds, a)[i] for i in idx])
    s.is_random = is_random
    return s


def make_solver(args, n_train):
    args.t_total = args.epochs * -(-n_train // args.batch_size)
    solver = MyFrame(CoRE_Net, dice_bce_loss, args, evalmode=True, pointmode=False)
    solver.load(args.pretrained)
    return solver


def evaluate(solver, ds, bs, thr):
    solver.eval()
    P, G = [], []
    with torch.no_grad():
        for img, m, b in DataLoader(ds, batch_size=bs):
            solver.set_input(img.cuda(), m, b)
            _, p = solver.test()
            P += [x.squeeze().cpu().numpy() for x in p]
            G += [x.squeeze().numpy() for x in m]
    return calculate_IoU_Dice(P, G, logger=log, threshold=thr)[1]['mDice']


def train(args, tr_ds, va_ds=None, ckpt=None):
    """Fine-tune. With va_ds+ckpt, keeps the best-val-mDice weights (pretrained is the baseline)."""
    solver = make_solver(args, len(tr_ds))
    best = None
    if va_ds is not None:
        best = evaluate(solver, va_ds, args.batch_size, args.threshold)
        solver.save(ckpt)
        print('pretrained val mDice:', best)
    for ep in range(args.epochs):
        solver.train()
        for img, m, b in DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=4):
            solver.set_input(img.cuda(), m, b)
            solver.optimize()
        if va_ds is not None:
            d = evaluate(solver, va_ds, args.batch_size, args.threshold)
            print('epoch', ep, 'val mDice', d)
            if d > best:
                best = d
                solver.save(ckpt)
    return solver, best


def main(args):
    
    rand = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))
    args.out_dir = osp.join(args.out_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '-' + rand)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(osp.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    full = ImageFolder(args.data_root, args.dataset, mode='test', is_random=False)
    groups = [key(n) for n in full.image_names]
    bs = args.batch_size

    if args.mode == 'split':
        leaves = sorted(set(groups))
        random.Random(args.seed).shuffle(leaves)
        n = len(leaves)
        a, b = int(args.train_frac * n), int((args.train_frac + args.val_frac) * n)
        sets = {'train': set(leaves[:a]), 'val': set(leaves[a:b]), 'final': set(leaves[b:])}
        ds = {k: subset(full, [i for i, g in enumerate(groups) if g in v], k == 'train')
              for k, v in sets.items()}
        for k in ds:
            print(k, len(sets[k]), 'leaves,', len(ds[k]), 'images')
        ckpt = osp.join(args.out_dir, 'ft_best.th')
        solver, _ = train(args, ds['train'], ds['val'], ckpt)
        solver.load(ckpt)
        print('final held-out mDice (run once):', evaluate(solver, ds['final'], bs, args.threshold))

    elif args.mode == 'cv':
        from sklearn.model_selection import GroupKFold
        idx = np.arange(len(groups))
        base, tuned = [], []
        for fold, (tr_i, va_i) in enumerate(GroupKFold(n_splits=args.folds).split(idx, groups=groups)):
            tr_ds, va_ds = subset(full, tr_i, True), subset(full, va_i, False)
            base.append(evaluate(make_solver(args, len(tr_ds)), va_ds, bs, args.threshold))
            solver, _ = train(args, tr_ds)           # fixed epochs, no selection on val
            tuned.append(evaluate(solver, va_ds, bs, args.threshold))
            solver.save(osp.join(args.out_dir, f'fold{fold}.th'))
            np.savetxt(osp.join(args.out_dir, 'cv_mdice.txt'), np.c_[base, tuned], header='pretrained fine-tuned')
            print('fold', fold, 'pretrained', base[-1], 'fine-tuned', tuned[-1])
        print('pretrained: %.4f | fine-tuned: %.4f +/- %.4f' % (np.mean(base), np.mean(tuned), np.std(tuned)))

    else:  # all
        solver, _ = train(args, subset(full, range(len(full.image_names)), True))
        path = osp.join(args.out_dir, 'ft_all.th')
        solver.save(path)
        print('saved', path)


if __name__ == '__main__':
    main(parse_args())