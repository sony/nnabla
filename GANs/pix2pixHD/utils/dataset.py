# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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
import fnmatch

image_extentions = [".png"]
file_type_id = {"leftImg8bit": 0, "instanceIds": 1, "labelIds": 2}


def get_cityscape_datalist(args, data_type="train", save_file=False):
    list_path = os.path.abspath("./data_list_{}.txt".format(data_type))
    if os.path.exists(list_path):
        with open(list_path, "r") as f:
            lines = f.readlines()

        return [line.strip().split(",") for line in lines]

    root_dir_path = os.path.abspath(args.data_dir)
    if not os.path.exists(root_dir_path):
        raise ValueError(
            "path for data_dir doesn't exist. ({})".format(args.data_dir))

    collections = {}
    for dirpath, dirnames, filenames in os.walk(root_dir_path):
        # really naive...
        if not fnmatch.fnmatch(dirpath, "*{}*".format(data_type)):
            continue

        images = [filename for filename in filenames if filename.endswith(
            *image_extentions)]

        if len(images) > 0:
            for image in images:
                key = "_".join(image.split("_")[:3])
                file_type = image.split("_")[-1].split(".")[0]

                if file_type not in file_type_id:
                    continue

                image_path = os.path.join(dirpath, image)
                if key not in collections:
                    collections[key] = [None, None, None]

                collections[key][file_type_id[file_type]] = image_path

    outs = collections.values()

    if save_file:
        write_outs = []
        for path_list in outs:
            if None in path_list:
                raise ValueError(
                    "unexpected error is happened during setting up dataset.")

            write_outs.append(",".join(path_list))

        with open(list_path, "w") as f:
            f.write("\n".join(write_outs))

    return list(outs)
