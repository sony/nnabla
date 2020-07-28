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
from collections import OrderedDict
from os.path import abspath, join, dirname, exists
from utils.common import check_update
from utils.type_conv import type_from_proto
import time
import yaml


def represent_odict(dumper, instance):
    return dumper.represent_mapping('tag:yaml.org,2002:map', instance.items())


yaml.add_representer(OrderedDict, represent_odict)

here = abspath(dirname(__file__))
base = abspath(join(here, '../..'))


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


def load_solver_info():
    solver_info = load_yaml_ordered(
        open(join(here, 'solvers.yaml'), 'r'))
    return solver_info


class MyDumper(yaml.Dumper):
    def __init__(self, *args, **kwargs):
        super(MyDumper, self).__init__(*args, **kwargs)

    def process_scalar(self):
        if self.analysis is None:
            self.analysis = self.analyze_scalar(self.event.value)
        if '\n' in self.analysis.scalar:
            self.style = '|'
        super(MyDumper, self).process_scalar()


def dump_yaml(data, stream, default_flow_style=None, width=None):
    dumper = MyDumper(stream, default_flow_style=default_flow_style, indent=2,
                      default_style=None, width=width)
    dumper.open()
    dumper.represent(data)
    dumper.close()


def info_to_list(info):
    '''Returns a list of (name, snake_name, [argument types as c++ type])'''
    items = []
    for name, item in info.items():
        items.append((name, item['snake_name'], [
            type_from_proto[v['type']]['cpp'] for v in item.get('arguments', {}).values()]))
    return items


def generate_function_types_one(template, function_name_camel, function_name_snake, ttypes_list, ext_info=None):
    kwargs = dict_filter(
        locals(), ['function_name_camel', 'function_name_snake', 'ttypes_list'])
    if ext_info is not None:
        kwargs.update(ext_info)
    generated = render_with_template(
        filename=template, template_kwargs=kwargs)
    return generated


def generate_function_types(function_info, function_types, ext_info=None, template=None, output_dir=None, output_format='%s.cpp'):
    if template is None:
        template = join(
            base, 'src/nbla/function/function.cpp.tmpl')
    if output_dir is None:
        output_dir = dirname(template)
    for name, ttypes_list in function_types.items():
        snake = function_info[name]['snake_name']
        generated = generate_function_types_one(
            template, name, snake, ttypes_list, ext_info=ext_info)
        path_o = join(output_dir, output_format % snake)
        check_update(path_o, generated, force=True)


def generate_solver_types_one(template, solver_name_camel, solver_name_snake, ttypes_list, ext_info=None):
    kwargs = dict_filter(
        locals(), ['solver_name_camel', 'solver_name_snake', 'ttypes_list'])
    if ext_info is not None:
        kwargs.update(ext_info)
    generated = render_with_template(
        filename=template, template_kwargs=kwargs)
    return generated


def generate_solver_types(solver_info, solver_types, ext_info=None, template=None, output_dir=None, output_format='%s.cpp'):
    if template is None:
        template = join(
            base, 'src/nbla/solver/solver.cpp.tmpl')
    if output_dir is None:
        output_dir = dirname(template)
    for name, ttypes_list in solver_types.items():
        snake = solver_info[name]['snake_name']
        generated = generate_solver_types_one(
            template, name, snake, ttypes_list, ext_info=ext_info)
        path_o = join(output_dir, output_format % snake)
        check_update(path_o, generated, force=True)


def generate_init(function_info, function_types, solver_info, solver_types, ext_info=None, template=None):
    if template is None:
        template = join(base, 'src/nbla/init.cpp.tmpl')
    # Create function list
    function_list = info_to_list(function_info)
    # Create solver list
    solver_list = info_to_list(solver_info)
    kwargs = dict_filter(
        locals(), ['function_list', 'function_types', 'solver_list', 'solver_types'])
    if ext_info is not None:
        kwargs.update(ext_info)
    generate_from_template(template, **kwargs)


def generate_version(template, rootdir, version = None, suffix=None, **kwargs):
    if suffix is not None:
        version = version + suffix
    tmpl_kwargs = dict(
        version=version, build_number=time.strftime('%y%m%d%H%M%S', time.gmtime()))
    tmpl_kwargs.update(kwargs)
    generated = render_with_template(filename=template, template_kwargs=tmpl_kwargs)
    path_o = template.replace('.tmpl', '')
    check_update(path_o, generated, force=True)


def unique_ordered(*lists):
    ret = []
    for l in lists:
        for v in l:
            if v not in ret:
                ret.append(v)
    return ret


def generate_skeleton_function_impl_one(ext_info, name, func, template, output_dir, output_format):
    path_o = join(output_dir, output_format % func['snake_name'])
    if exists(path_o):
        return
    in_types = [v.get('template', 'T')
                for v in func['inputs'].values()]
    out_types = [v.get('template', 'T')
                 for v in func['outputs'].values()]
    ttypes = unique_ordered(in_types, out_types)
    if 'arguments' not in func:
        func['arguments'] = {}
    generated = render_with_template(filename=template, template_kwargs=dict_union(
        ext_info, dict(name=name, in_types=in_types, out_types=out_types, ttypes=ttypes, **func)))
    check_update(path_o, generated, force=False)


def generate_skeleton_function_impl(function_info, function_types, ext_info={}, template=None, output_dir=None, output_format='%s.cpp'):
    if template is None:
        template = join(
            base, 'src/nbla/function/generic/function_impl.cpp.tmpl')
    if output_dir is None:
        output_dir = dirname(template)

    for name, func in function_info.items():
        if name not in function_types:
            continue
        generate_skeleton_function_impl_one(
            ext_info, name, func, template, output_dir, output_format)


def generate_skeleton_backward_function_impl(function_info, template, output_dir, output_format='%s.py'):
    """This function now generate the template of a backward function in python-layer using PythonFunction.
    """
    from mako.template import Template
    import os

    for name, func in function_info.items():
        path_o = join(output_dir, output_format % func['snake_name'])
        if os.path.exists(path_o):
            continue
        # Create the skelton file of the backward function
        generated = render_with_template(
            filename=template, template_kwargs=dict(func=func))
        check_update(path_o, generated, force=False)
