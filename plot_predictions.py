#!/usr/bin/env python3
"""Plot image / ground truth / probability / binary prediction / error map for a checkpoint."""
import argparse
import json
import os.path as osp
import random
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from eval_finetuned import get_probs
from finetune import subset
from utils.data_selftrain import ImageFolder


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--eval-root', default=None, help='default: data_root from config.json')
    p.add_argument('--ckpt', default=None, help='default: <run-dir>/ft_all.th')
    p.add_argument('--threshold', default=0.91, type=float)
    p.add_argument('--n', default=8, type=int)
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--out', default=None, help='default: <run-dir>/predictions.png')
    a = p.parse_args()

    args = SimpleNamespace(**json.load(open(osp.join(a.run_dir, 'config.json'))))
    ckpt = a.ckpt or osp.join(a.run_dir, 'ft_all.th')
    
    root = a.eval_root or args.data_root
    name = f"{osp.basename(osp.normpath(root))}_n{a.n}_s{a.seed}_predictions.png"
    out = a.out or osp.join(a.run_dir, name)

    ds = ImageFolder(a.eval_root or args.data_root, args.dataset, mode='test', is_random=False)
    n = min(a.n, len(ds))
    idx = sorted(random.Random(a.seed).sample(range(len(ds)), n))   # fixed random sample, not cherry-picked
    sub = subset(ds, idx, False)
    P, G = get_probs(args, ckpt, sub)

    fig, axes = plt.subplots(n, 5, figsize=(7.5, 1.5 * n), squeeze=False)
    for i in range(n):
        img = sub[i][0].permute(1, 2, 0).numpy()
        rgb = ((img + 1.6) / 3.2)[..., ::-1].clip(0, 1)            # cv2 loads BGR
        pred, gt = P[i] >= a.threshold, G[i] > 0.5
        dice = 2 * (pred & gt).sum() / max(pred.sum() + gt.sum(), 1)
        err = np.zeros(pred.shape + (3,))
        err[pred & gt] = [1, 1, 1]        # true positive: white
        err[pred & ~gt] = [1, 0, 0]       # false positive: red
        err[~pred & gt] = [0, 0.4, 1]     # false negative: blue

        panels = [(rgb, sub.image_names[i][-28:], None), (gt, 'ground truth', 'gray'),
                  (P[i], 'probability', 'gray'), (pred, 'pred >= %.2f' % a.threshold, 'gray'),
                  (err, 'dice %.3f (red FP, blue FN)' % dice, None)]
        for ax, (im, title, cmap) in zip(axes[i], panels):
            ax.imshow(im, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
            ax.set_title(title, fontsize=8)
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print('saved', out)


if __name__ == '__main__':
    main()