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


from __future__ import print_function

from os.path import abspath, join, dirname

import yaml
from collections import OrderedDict


def represent_odict(dumper, instance):
    return dumper.represent_mapping('tag:yaml.org,2002:map', instance.items())


yaml.add_representer(OrderedDict, represent_odict)

here = abspath(dirname(__file__))


def load_yaml_ordered(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    '''
    Load function with keeping the order of dictionaries.
    '''
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


def render_with_template(text=None, filename=None, preprocessor=None, template_kwargs={}):
    from mako.template import Template
    from mako import exceptions

    tmpl = Template(text=text, filename=filename, preprocessor=preprocessor)
    try:
        return tmpl.render(**template_kwargs)
    except Exception as e:
        import sys
        print('-' * 78, file=sys.stderr)
        print('Template exceptions', file=sys.stderr)
        print('-' * 78, file=sys.stderr)
        print(exceptions.text_error_template().render(), file=sys.stderr)
        print('-' * 78, file=sys.stderr)
        raise e


def write_generated(filename, generated):
    with open(filename, 'wb') as f:
        write_content = generated.encode('utf_8')
        write_content = write_content.replace(b'\r\n', b'\n')
        write_content = write_content.replace(b'\r', b'\n')
        f.write(write_content)


def dict_union(*dicts):
    ret = {}
    for d in dicts:
        ret.update(d)
    return ret


def dict_filter(d, keys):
    return {key: d[key] for key in keys}


def generate_from_template(path_template, overwrite=True, **kwargs):
    from utils.common import check_update
    # ^ Need nnabla/build-tools/code_generator/ in python path.

    path_out = path_template.replace('.tmpl', '')
    generated = render_with_template(
        filename=path_template, template_kwargs=kwargs)
    check_update(path_out, generated, force=overwrite)


def load_function_info(flatten=False):
    ret = load_yaml_ordered(open(join(here, 'functions.yaml'), 'r'))
    if not flatten:
        return ret
    ret2 = OrderedDict()
    for _, cat in ret.items():
        for func_name, func in cat.items():
            ret2[func_name] = func
    return ret2


class MyDumper(yaml.Dumper):
    def __init__(self, *args, **kwargs):
        super(MyDumper, self).__init__(*args, **kwargs)

    def process_scalar(self):
        if self.analysis is None:
            self.analysis = self.analyze_scalar(self.event.value)
        if '\n' in self.analysis.scalar:
            self.style = '|'
        super(MyDumper, self).process_scalar()


def dump_yaml(data, stream, default_flow_style=None):
    dumper = MyDumper(stream, default_flow_style=default_flow_style, indent=2,
                      default_style=None)
    dumper.open()
    dumper.represent(data)
    dumper.close()
