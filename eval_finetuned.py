#!/usr/bin/env python3
"""Evaluate checkpoints saved by finetune.py, using the run dir's config.json.

cv    : each fold{k}.th on its own held-out leaves (split re-created), plus the pretrained baseline
split : ft_best.th on the 'final' leaves (and 'val'), plus the pretrained baseline
all   : ft_all.th on all labeled images (training data, so only a sanity check)
"""
import argparse
import json
import os.path as osp
import random
from types import SimpleNamespace

import glog as log
import numpy as np
import torch
from torch.utils.data import DataLoader

from finetune import key, make_solver, subset
from utils.data_selftrain import ImageFolder
from utils.metrics import calculate_IoU_Dice


def get_probs(args, ckpt, ds):
    args.pretrained = ckpt                    # make_solver loads args.pretrained
    solver = make_solver(args, len(ds))
    solver.eval()
    P, G = [], []
    with torch.no_grad():
        for img, m, b in DataLoader(ds, batch_size=args.batch_size):
            solver.set_input(img.cuda(), m, b)
            _, p = solver.test()
            P += [x.squeeze().cpu().numpy() for x in p]
            G += [x.squeeze().numpy() for x in m]
    return P, G


def score(P, G, thrs):
    return [calculate_IoU_Dice(P, G, logger=log, threshold=t)[1]['mDice'] for t in thrs]


def report(name, P, G, thrs):
    print(name.ljust(22), '  '.join('%.2f: %.4f' % (t, s) for t, s in zip(thrs, score(P, G, thrs))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--thresholds', default='0.5,0.7,0.9', type=str)
    p.add_argument('--eval-root', default=None,
                   help='data root holding unseen labeled leaves: <eval-root>/<dataset>/test/<leaf>/...')
    a = p.parse_args()
    thrs = [float(t) for t in a.thresholds.split(',')]

    cfg = json.load(open(osp.join(a.run_dir, 'config.json')))
    args = SimpleNamespace(**cfg)
    pretrained = cfg['pretrained']

    full = ImageFolder(args.data_root, args.dataset, mode='test', is_random=False)
    groups = [key(n) for n in full.image_names]

    if a.eval_root:
        ev = ImageFolder(a.eval_root, args.dataset, mode='test', is_random=False)
        overlap = sorted({key(n) for n in ev.image_names} & set(groups))
        print(len(ev.image_names), 'images;', len(overlap), 'leaf keys also in training data:', overlap)
        ckpts = {'pretrained': pretrained}
        if args.mode == 'cv':
            ckpts.update({'fold%d' % k: osp.join(a.run_dir, 'fold%d.th' % k) for k in range(args.folds)})
        else:
            ckpts['fine-tuned'] = osp.join(a.run_dir, 'ft_all.th' if args.mode == 'all' else 'ft_best.th')
        for name, c in ckpts.items():
            report(name, *get_probs(args, c, ev), thrs)
        return

    if args.mode == 'cv':
        from sklearn.model_selection import GroupKFold
        for fold, (tr_i, va_i) in enumerate(GroupKFold(n_splits=args.folds).split(np.arange(len(groups)), groups=groups)):
            va = subset(full, va_i, False)
            report('fold%d pretrained' % fold, *get_probs(args, pretrained, va), thrs)
            report('fold%d fine-tuned' % fold, *get_probs(args, osp.join(a.run_dir, 'fold%d.th' % fold), va), thrs)

    elif args.mode == 'split':
        leaves = sorted(set(groups))
        random.Random(args.seed).shuffle(leaves)
        n = len(leaves)
        i, j = int(args.train_frac * n), int((args.train_frac + args.val_frac) * n)
        for name, ks in [('val', leaves[i:j]), ('final', leaves[j:])]:
            ds = subset(full, [k for k, g in enumerate(groups) if g in set(ks)], False)
            report(name + ' pretrained', *get_probs(args, pretrained, ds), thrs)
            report(name + ' fine-tuned', *get_probs(args, osp.join(a.run_dir, 'ft_best.th'), ds), thrs)

    else:
        ds = subset(full, range(len(groups)), False)
        report('all pretrained', *get_probs(args, pretrained, ds), thrs)
        report('all fine-tuned (train)', *get_probs(args, osp.join(a.run_dir, 'ft_all.th'), ds), thrs)


if __name__ == '__main__':
    main()