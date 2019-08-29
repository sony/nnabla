# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import argparse
import glob
import json
from itertools import islice
import shutil
from collections import defaultdict
import numpy as np


def create_image_id(args, data):
    annotation_categories = data['categories']
    # Obtain {name: id} table (Note: the original COCO category IDs are not continuous)
    name_to_id = dict(
        map(lambda x: (x['name'], x['id']), annotation_categories))
    # Obtain {name: id80} table (Note: Creating continuous IDS from 0 to 79
    # IDs are assigned in ascending order of the original IDs.
    name_to_id80 = dict([(name, id80) for id80, name in enumerate(
        map(lambda x: x['name'], annotation_categories))])
    # Obtain {id80: subid}
    # (Sequential IDs for selected categories.
    # Used to convert IDs in darknet label file to sequential IDS of the selected categories)
    list_id80_selected = [name_to_id80[name] for name, id_orig in name_to_id.items(
    ) if id_orig in args.selected_classes]
    id80_to_subid = {str(id80): id_sel for id_sel,
                     id80 in enumerate(sorted(list_id80_selected))}
    return name_to_id, name_to_id80, id80_to_subid


def create_image_list(args, N, which):
    annotation_path = os.path.join(
        args.data_path, 'annotations/instances_' + which + '2014.json')
    data_json = json.load(open(annotation_path, 'r'))
    annotation_data = data_json['annotations']
    images_by_class = defaultdict(set)
    for data in annotation_data:
        class_id = data['category_id']
        if class_id in args.selected_classes:
            images_by_class[class_id].add(data['image_id'])
    # Obtain N files per categories
    image_subset = set()
    for ids in args.selected_classes:
        image_subset.update(list(images_by_class[ids])[:N])
    # Save a list of files
    image_subset = list(map(lambda x: args.data_path+'/images/'+which +
                            '2014/COCO_'+which+'2014_' + '%012d.jpg' % int(x), image_subset))
    path = os.path.join(args.subset_path, which+'2014.txt')
    with open(path, 'a+') as fo:
        for im in image_subset:
            fo.writelines('%s\n' % im)
    return data_json
    print("################ completed ################")


def create_labels(args, id80_to_subid, which):
    image_files = np.loadtxt(args.subset_path+which+'2014'+'.txt', dtype=str)
    for image_file in image_files:
        name = os.path.splitext(os.path.basename(image_file))[0]
        label_file = os.path.join(
            args.data_path, 'labels-original', which+'2014', name + '.txt')
        labels = np.loadtxt(label_file, ndmin=2, dtype='str').tolist()
        # Filter categories not selected
        sublabels = [x for x in labels if (
            np.isin(x[0], list(id80_to_subid.keys())))]
        # convert IDs
        sub_labels = list(
            map(lambda x: [str(id80_to_subid[str(x[0])])] + x[1:], sublabels))
        if not os.path.exists(os.path.join(args.data_path, 'labels', which+'2014')):
            os.mkdir(os.path.join(args.data_path, 'labels', which+'2014'))
        label_out_path = os.path.join(
            args.data_path, 'labels', which+'2014', name+'.txt')
        np.savetxt(label_out_path, sub_labels, fmt=' '.join(['%s'] * 5))
    print("################ completed ################")


def coco_names_subset(args, name_to_id80, id80_to_subid):
    print("########## creating coco.names ##########")
    # Make an inverse map for name
    id80_to_name = dict((v, k) for k, v in name_to_id80.items())
    # Obtain category names in a sorted order in 80 categories (0-78)
    # Note sorting is required since Python < 3.6 doesn't ensure the key order.
    id80_to_subid = dict((int(k), v) for k, v in id80_to_subid.items())
    subnames = [id80_to_name[int(id80)]
                for id80 in sorted(id80_to_subid.keys())]
    np.savetxt(args.subset_path + '/coco.names', subnames, fmt='%s')
    print("################ completed ################")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data-path', type=str,
                        help='Path to coco dataset', default='/darknet/data/coco')
    parser.add_argument('-sp', '--subset-path', type=str, help='Path for generating subset coco dataset',
                        default='./subset')
    parser.add_argument('-it', '--images-train', type=int,
                        help='no. of images in each training class', default=100)
    parser.add_argument('-iv', '--images-val', type=int,
                        help='no. of images in each validation class', default=50)
    parser.add_argument('-l', '--selected-classes', nargs='+',
                        type=int, help='select the desired class ids', required=True)
    argdict = parser.parse_args()
    return argdict


def main():
    args = parse_args()
    print("#########creating training images##########")
    train_images = create_image_list(args, args.images_train, 'train')
    print("###########creating val images ############")
    val_images = create_image_list(args, args.images_val, 'val')
    name_to_id, name_to_id80, id80_to_subid = create_image_id(
        args, train_images)
    print("#########creating training labels##########")
    train_labels = create_labels(args, id80_to_subid, 'train')
    print("###########creating val labels ############")
    val_labels = create_labels(args, id80_to_subid, 'val')
    category_names = coco_names_subset(args, name_to_id80, id80_to_subid)


if __name__ == "__main__":
    main()
