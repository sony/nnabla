###############################################################################
# Code from
# https://github.com/NVIDIA/pix2pixHD/blob/master/util/html.py
# Modified so it fits nnabla usage more.
###############################################################################

import dominate
from dominate.tags import *
import os


def check_dir_exist_and_create(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class HtmlCreater(object):
    def __init__(self, root, page_title="no title", redirect_interval=0):
        self.root = root
        self.img_dir = os.path.join(os.path.abspath(root), "imgs")

        check_dir_exist_and_create(self.img_dir)

        self.doc = dominate.document(title=page_title)

        if redirect_interval > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(redirect_interval))

    def add_text(self, text):
        with self.doc:
            h3(text)

    def add_images(self, imgs, im_titles, width=512):
        t = table(border=1, style="table-layout: fixed;")
        self.doc.add(t)
        with t:
            for im, title in zip(imgs, im_titles):
                href = os.path.join("imgs", im)
                with td(style="overflow-wrap: break-word", align="center", valign="top"):
                    with p():
                        with a(href=href):
                            img(style="width:{}px".format(width), src=href)

                        br()
                        p(title)

    def save(self):
        with open(os.path.join(self.root, "index.html"), "w") as f:
            f.write(str(self.doc))
