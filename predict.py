#!/usr/bin/env python3
"""Predict probability masks for images in a folder; save <name>_prob.png next to each image (16-bit)."""
import argparse
import json
import os.path as osp
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from finetune import make_solver


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True, help='finetune run folder (config.json + checkpoint)')
    p.add_argument('--image-dir', required=True)
    p.add_argument('--ckpt', default=None, help='default: <run-dir>/ft_all.th')
    p.add_argument('--pattern', default='*.jpg', help="glob, e.g. '*.jpg' or '*leaf*'")
    p.add_argument('--recursive', action='store_true')
    p.add_argument('--model-size', action='store_true', help='keep 448x448 instead of resizing back to the photo size')
    a = p.parse_args()

    args = SimpleNamespace(**json.load(open(osp.join(a.run_dir, 'config.json'))))
    args.pretrained = a.ckpt or osp.join(a.run_dir, 'ft_all.th')
    solver = make_solver(args, 1)
    solver.eval()
    solver.net.mask_point_on = False

    root = Path(a.image_dir)
    paths = sorted(root.rglob(a.pattern) if a.recursive else root.glob(a.pattern))
    paths = [q for q in paths if not q.stem.endswith('_prob')]
    print(len(paths), 'images')

    bs = args.batch_size
    for i in range(0, len(paths), bs):
        chunk, imgs = [], []
        for q in paths[i:i + bs]:
            im = cv2.imread(str(q))
            if im is None:
                print('could not read', q)
                continue
            chunk.append(q)
            imgs.append(im)
        if not imgs:
            continue
        # same preprocessing as image_reader / default_Dataset_loader
        x = np.stack([cv2.resize(im, (448, 448), interpolation=cv2.INTER_NEAREST)
                      .astype(np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6 for im in imgs])
        with torch.no_grad():
            out = solver.net.forward(torch.from_numpy(x).cuda())['mask_coarse_logits']
            out = out.sigmoid().squeeze(1).cpu().numpy()
        for q, im, pr in zip(chunk, imgs, out):
            if not a.model_size:
                pr = cv2.resize(pr, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(q.with_name(q.stem + '_prob.png')), (pr * 65535).round().astype(np.uint16))
    print('done')


if __name__ == '__main__':
    main()