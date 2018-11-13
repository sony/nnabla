# This code has been forked from https://github.com/CasiaFan/Dataset_to_VOC_converter/blob/master/anno_coco2voc.py.
# For usage and license agreements visit the given website
import argparse
import json
import cytoolz
from lxml import etree, objectify
import os
import re


def instance2xml_base(anno):
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('VOC2014_instance/{}'.format(anno['category_id'])),
        E.filename(anno['file_name']),
        E.source(
            E.database('MS COCO 2014'),
            E.annotation('MS COCO 2014'),
            E.image('Flickr'),
            E.url(anno['coco_url'])
        ),
        E.size(
            E.width(anno['width']),
            E.height(anno['height']),
            E.depth(3)
        ),
        E.segmented(0),
    )
    return anno_tree


def instance2xml_bbox(anno, bbox_type='xyxy'):
    """bbox_type: xyxy (xmin, ymin, xmax, ymax); xywh (xmin, ymin, width, height)"""
    assert bbox_type in ['xyxy', 'xywh']
    if bbox_type == 'xyxy':
        xmin, ymin, w, h = anno['bbox']
        xmax = xmin+w
        ymax = ymin+h
    else:
        xmin, ymin, xmax, ymax = anno['bbox']
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.object(
        E.name(anno['category_id']),
        E.bndbox(
            E.xmin(xmin),
            E.ymin(ymin),
            E.xmax(xmax),
            E.ymax(ymax)
        ),
        E.difficult(anno['iscrowd'])
    )
    return anno_tree


def parse_instance(content, outdir):
    categories = {d['id']: d['name'] for d in content['categories']}
    # merge images and annotations: id in images vs image_id in annotations
    merged_info_list = list(map(cytoolz.merge, cytoolz.join(
        'id', content['images'], 'image_id', content['annotations'])))
    # convert category id to name
    for instance in merged_info_list:
        instance['category_id'] = categories[instance['category_id']]
    # group by filename to pool all bbox in same file
    for name, groups in cytoolz.groupby('file_name', merged_info_list).items():
        anno_tree = instance2xml_base(groups[0])
        for group in groups:
            anno_tree.append(instance2xml_bbox(group, bbox_type='xyxy'))
        filename = os.path.join(outdir, os.path.splitext(name)[0] + ".xml")
        etree.ElementTree(anno_tree).write(filename, pretty_print=True)
        print("Formating instance xml file {} done!".format(name))


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    content = json.load(open(args.anno_file, 'r'))
    parse_instance(content, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--anno_file", help="annotation file for object instance/keypoint")
    parser.add_argument(
        "--output_dir", help="output directory for voc annotation xml file")
    args = parser.parse_args()
    main(args)
