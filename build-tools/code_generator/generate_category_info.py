import json
import sys
import zlib
from collections import OrderedDict

from utils.load_function_rst import Functions

info = OrderedDict()
f = Functions()

all_functions = f.info['Functions']

info['categories'] = OrderedDict()

for category, functions in all_functions.items():
    if category not in info:
        info['categories'][category] = OrderedDict()
    for function, function_info in functions.items():
        func = OrderedDict()
        func['name'] = function
        func['snakecase_name'] = f.info['Names'][function]
        func['input'] = OrderedDict()
        for n, i in function_info['input'].items():
            func['input'][n] = OrderedDict()
            if 'Options' in i:
                func['input'][n]['Options'] = i['Options'].split()

        if 'argument' in function_info:
            func['argument'] = OrderedDict()
            for n, i in function_info['argument'].items():
                func['argument'][n] = OrderedDict()
                func['argument'][n]['Type'] = i['Type']
                if 'TypeSelection' in i:
                    func['argument'][n]['TypeSelection'] = i['TypeSelection']

        func['output'] = OrderedDict()
        for n, i in function_info['output'].items():
            func['output'][n] = OrderedDict()
            if 'Options' in i:
                func['output'][n]['Options'] = i['Options'].split()
        info['categories'][category][function] = func

info['version'] = zlib.crc32(json.dumps(
    info['categories']).encode('utf-8')) & 0x7ffffff

with open(sys.argv[1], 'w') as f:
    f.write(json.dumps(info, indent=4))
