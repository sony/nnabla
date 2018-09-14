import os
import sys
import re
import fnmatch

# from https://github.com/woodylee1974/simple_sm by Woody
class StateMachine():
    def __init__(self, name, handler, **kwargs):
        self.transit_map_ = {}
        self.name_ = name
        self.current_state_ = None
        self.handler_ = handler
        self.events_ = set()
        self.event = None
        self.__dict__.update(kwargs)
        self.next_state = None
        self.default_handle = None
        if 'start' in kwargs:
            self.current_state_ = kwargs['start']
        self.parser_ = re.compile(\
            r'^([^ \t\n\r\f\v\->]*)[\s]*[\-]+[>]?[\s]*([^ \t\n\r\f\v\->]*)[\s]*[\-]+>[\s]*([^ \t\n\r\f\v\->]*)$')
        cls = handler.__class__
        for k, v in cls.__dict__.items():
            if hasattr(v, '__call__') and v.__doc__ is not None:
                self._add_transit_by(v, v.__doc__)
    
    def _event_func(self, *args, **kwargs):
        self.handle_event(self.event, *args, **kwargs)

    def _add_transit_by(self, v, trans):
        for tran in trans.split('\n'):
            tran = tran.strip()
            trans_line = self.parser_.match(tran)
            if trans_line:
                self.add_transit(trans_line.group(1), trans_line.group(2), \
                                trans_line.group(3), v)
                if self.current_state_ is None:
                    self.current_state_ = trans_line.group(1)
                self.events_.add(trans_line.group(2))
            elif tran.strip() == 'default_handle':
                self.default_handle = v

    def __getattr__(self, item):
        for event in self.events_:
            if fnmatch.fnmatch(item, event):
                self.event = item
                return self._event_func

    def add_transit(self, s0, e, s1, func=None):
        if s0 in self.transit_map_:
            handles = self.transit_map_[s0]
            handles[e] = {'func': func, 'state': s1}
        else:
            self.transit_map_[s0] = {e: {'func': func, 'state': s1}}

    def start_state(self, s):
        self.current_state_ = s

    def handle_event(self, e, *args, **kwargs):
        handled = False
        self.handler_.current_event = e
        if self.current_state_ in self.transit_map_:
            handles = self.transit_map_[self.current_state_]
            for k, trans in handles.items():
                if fnmatch.fnmatch(e, k):
                    func = trans['func']
                    self.next_state = handles[k]['state']
                    ret = func(self.handler_, *args, **kwargs)
                    current_state = self.current_state_
                    transit_done = True
                    if ret is None:
                        self.current_state_= self.next_state
                    elif ret == True:
                        self.current_state_= self.next_state
                    else:
                        transit_done = False
                    handled = True
                    if self.debug:
                        if transit_done:
                            print("[%s][%s -- %s --> %s]" % (self.name_,
                                                                 current_state,
                                                                 e,
                                                                 self.current_state_))
                        else:
                            print("[%s][%s -- %s --> %s[%s]][Transition is refused]" % (self.name_,
                                                                 current_state,
                                                                 e,
                                                                 self.current_state_,
                                                                 self.next_state))
                        # for a in args:                                
                        #     print(a)
                        # for k, v in kwargs.items():
                        #     print('%s=%o' %(k,v))
        if not handled:
            if self.debug:
                print("[%s][%s -- %s <-- %s]" % (self.name_,
                                                        self.current_state_,
                                                        e,
                                                        'not handled'))
            if self.default_handle:
                self.default_handle(self.handler_, *args, **kwargs)

    def get_state(self):
        return self.current_state_
        
    def set_next_state(self, next_state):
        self.next_state = next_state

    def dump(self):
        for (s, v) in self.transit_map_.items():
            print(s, v)


class StatusHandler:
    def __init__(self, output_buffer, test_result):
        self.output_buffer = output_buffer
        self.test_result = test_result
        self.count = 0
        self.ok = 0
        self.count_line = -1

    def parse_upper_line(self, input_line):
        'start -- equal_line --> upper_line_found'
        self.output_buffer += [input_line]
    
    def parse_operator(self, input_line):
        'upper_line_found -- accept_operator --> operator_found'
        self.output_buffer += [input_line]
    
    def parse_lower_line(self, input_line):
        'operator_found -- equal_line --> import_table'
        self.field_lens = [len(f) + 1 for f in input_line.split(' ')]
        self.output_buffer += [input_line]

    def handle_import_table(self, input_line):
        'import_table --> process_line --> import_table'
        output_line = '{{:<{}}}{{:<{}}}{{:<{}}}'.format(*self.field_lens)
        if input_line[0] != ' ':
            fields = filter(lambda x: x != '', input_line.split('  '))
            fields = [f.strip() for f in fields]
            self.count += 1
            if fields[0] in self.test_result:
                if self.test_result[fields[0]] == 'OK':
                    self.ok += 1
                line = output_line.format(fields[0], self.test_result[fields[0]], ' '.join(fields[2:]))
                self.output_buffer += [line]
                self.output_buffer += '\n'
            else:
                line = output_line.format(fields[0], 'Not test', ' '.join(fields[2:]))
                self.output_buffer += [line]
                self.output_buffer += '\n'
        else:
            self.output_buffer += [input_line]
    
    def default_handle(self, input_line):
        '''
        default_handle
        finish_table-->process_line-->finish_table
        '''
        self.output_buffer += [input_line]
    
    def finish_import_table(self, input_line):
        'import_table --> equal_line --> finish_table'
        self.output_buffer += [input_line]
        if self.count_line > 0:
            self.output_buffer[self.count_line] = 'Count {}/{}\n'.format(self.ok, self.count)
        self.count_line = -1
    
    def process_count(self, input_line):
        'start --> accept_count --> start'
        self.count_line = len(self.output_buffer)
        self.output_buffer += ['']
        self.count = 0
        self.ok = 0



CURRENT_PATH=os.path.dirname(__file__)
TEMPALTE_FILE=os.path.join(CURRENT_PATH, 'onnx_test_report.rst.tmpl')
OUTPUT_FILE=os.path.join(CURRENT_PATH, '../../../../doc/python/file_format_converter/onnx/operator_coverage.rst')

def gen_report(import_result, export_result):
    with open(TEMPALTE_FILE, 'r') as f:
        line_buffer = []
        importer_status_handler = StateMachine('ImporterSM',
                                        StatusHandler(line_buffer, import_result),
                                        start='start', debug=False)
        exporter_status_handler = StateMachine('ExporterSM',
                                        StatusHandler(line_buffer, export_result),
                                        start='start', debug=False)
        for line in f.readlines():
            field = line[:8]
            if exporter_status_handler.get_state() == 'finish_table':
                exporter_status_handler.start_state('start')            
            if importer_status_handler.get_state() != 'finish_table':
                if field == '=' * 8:
                    importer_status_handler.equal_line(line)
                elif field == 'Operator':
                    importer_status_handler.accept_operator(line)
                else:
                    importer_status_handler.process_line(line)
            else: 
                if field == '=' * 8:
                    exporter_status_handler.equal_line(line)
                elif field == 'Operator':
                    exporter_status_handler.accept_operator(line)
                elif field[:5] == 'Count':
                    exporter_status_handler.accept_count(line)
                else:
                    exporter_status_handler.process_line(line)
        with open(OUTPUT_FILE, 'w') as of:
            line_buffer = ''.join(line_buffer)
            of.write(line_buffer)
            print('\n{} is updated.'.format(os.path.basename(OUTPUT_FILE)))



