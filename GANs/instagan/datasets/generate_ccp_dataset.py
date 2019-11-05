# Copyright (c) 2018, Sangwoo Mo
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# This code is taken from https://github.com/sangwoomo/instagan (b67e900).
# The source code is provided under a license described above.


import argparse
import numpy as np
import scipy.io as sio
from pathlib import Path
from tqdm import tqdm
from PIL import Image


def main():
	parser = create_argument_parser()
	args = parser.parse_args()
	generate_ccp_dataset(args)

def create_argument_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_root', type=str, default='datasets/clothing-co-parsing')
	parser.add_argument('--save_root', type=str, default='datasets/jeans2skirt_ccp')
	parser.add_argument('--cat1', type=str, default='jeans', help='category 1')
	parser.add_argument('--cat2', type=str, default='skirt', help='category 2')
	return parser

def generate_ccp_dataset(args):
	"""Generate COCO dataset (train/val, A/B)"""
	args.data_root = Path(args.data_root)
	args.img_root = args.data_root / 'photos'
	args.pix_ann_root = args.data_root / 'annotations' / 'pixel-level'
	args.img_ann_root = args.data_root / 'annotations' / 'image-level'
	args.pix_ann_ids = get_ann_ids(args.pix_ann_root)
	args.img_ann_ids = get_ann_ids(args.img_ann_root)

	args.label_list = sio.loadmat(str(args.data_root / 'label_list.mat'))['label_list'].squeeze()

	args.save_root = Path(args.save_root)
	args.save_root.mkdir()

	generate_ccp_dataset_train(args, 'A', args.cat1)
	generate_ccp_dataset_train(args, 'B', args.cat2)
	generate_ccp_dataset_val(args, 'A', args.cat1)
	generate_ccp_dataset_val(args, 'B', args.cat2)

def generate_ccp_dataset_train(args, imset, cat):
	img_path = args.save_root / 'train{}'.format(imset)
	seg_path = args.save_root / 'train{}_seg'.format(imset)
	img_path.mkdir()
	seg_path.mkdir()

	cat_id = get_cat_id(args.label_list, cat)

	pb = tqdm(total=len(args.pix_ann_ids))
	pb.set_description('train{}'.format(imset))
	for ann_id in args.pix_ann_ids:
		ann = sio.loadmat(str(args.pix_ann_root / '{}.mat'.format(ann_id)))['groundtruth']
		if np.isin(ann, cat_id).sum() > 0:
			img = Image.open(args.img_root / '{}.jpg'.format(ann_id))
			img.save(img_path / '{}.png'.format(ann_id))
			seg = (ann == cat_id).astype('uint8')  # get segment of given category
			seg = Image.fromarray(seg * 255)
			seg.save(seg_path / '{}_0.png'.format(ann_id))
		pb.update(1)
	pb.close()

def generate_ccp_dataset_val(args, imset, cat):
	img_path = args.save_root / 'val{}'.format(imset)
	seg_path = args.save_root / 'val{}_seg'.format(imset)
	img_path.mkdir()
	seg_path.mkdir()

	cat_id = get_cat_id(args.label_list, cat)

	pb = tqdm(total=len(args.img_ann_ids))
	pb.set_description('val{}'.format(imset))
	for ann_id in args.img_ann_ids:
		ann = sio.loadmat(str(args.img_ann_root / '{}.mat'.format(ann_id)))['tags']
		if np.isin(ann, cat_id).sum() > 0:
			img = Image.open(args.img_root / '{}.jpg'.format(ann_id))
			img.save(img_path / '{}.png'.format(ann_id))
		pb.update(1)
	pb.close()

def get_ann_ids(anno_path):
	ids = list()
	for p in anno_path.iterdir():
		ids.append(p.name.split('.')[0])
	return ids

def get_cat_id(label_list, cat):
	for i in range(len(label_list)):
		if cat == label_list[i][0]:
			return i

if __name__ == '__main__':
	main()
