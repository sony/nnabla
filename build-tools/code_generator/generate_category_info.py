import json
import sys
from collections import OrderedDict

from utils.load_function_rst import Functions

info = OrderedDict()
f = Functions()

all_functions = f.info['Functions']
for category, functions in all_functions.items():
    if category not in info:
        info[category] = OrderedDict()
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

        func['output'] = OrderedDict()
        for n, i in function_info['output'].items():
            func['output'][n] = OrderedDict()
            if 'Options' in i:
                func['output'][n]['Options'] = i['Options'].split()
        info[category][function] = func


with open(sys.argv[1], 'w') as f:
    f.write(json.dumps(info, indent=4))
