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

from os.path import abspath, dirname, join
from recommonmark.parser import CommonMarkParser
from recommonmark.transform import AutoStructify
import os

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon',
              'sphinx.ext.todo',
              'sphinx.ext.viewcode',
              'sphinx.ext.intersphinx',
              'sphinxcontrib.actdiag',
              'sphinxcontrib.blockdiag',
              'sphinxcontrib.nwdiag',
              'sphinxcontrib.plantuml',
              'sphinxcontrib.seqdiag',
              'sphinxcontrib.toc']

templates_path = ['_templates']

# source_parsers = {
#     '.md': CommonMarkParser,
# }
# source_suffix = ['.rst', '.md']

master_doc = 'index'

# General information about the project.
project = u'NNabla'
copyright = u'2017, Sony Corporation'
author = u'Sony Corporation'

version = u'0.9.1'
release = u'0.9.1rc3'

language = None

exclude_patterns = ['third_party']
pygments_style = 'sphinx'

todo_include_todos = True

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
nitpicky = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3.4', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
}


# At the bottom of conf.py
def setup(app):
    app.add_config_value('recommonmark_config', {
        'auto_toc_tree_section': 'Contents',
    }, True)
    app.add_transform(AutoStructify)
    app.add_stylesheet('custom.css')
