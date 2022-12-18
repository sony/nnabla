# Copyright 2022 Sony Group Corporation.
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


import inspect


def _extract_argument_signature(sigstr):
    return sigstr[:sigstr.rfind(')')].strip('()')


def _extract_return_annotation(sigstr, has_return_anno):
    if not has_return_anno:
        return ''
    return sigstr.split(')')[1]


def _create_caller_signature(sig):
    args = []
    for i, (k, v) in enumerate(sig.parameters.items()):
        if v.kind is inspect.Parameter.POSITIONAL_ONLY or v.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            args.append(k)
        elif v.kind is inspect.Parameter.VAR_POSITIONAL:
            args.append(str(v))
        elif v.kind is inspect.Parameter.KEYWORD_ONLY:
            args.append(f'{k}={k}')
        elif v.kind is inspect.Parameter.VAR_KEYWORD:
            args.append(str(v))
    return ', '.join(args)


P = inspect.Parameter
PARAM_KINDS = [P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD,
               P.VAR_POSITIONAL, P.KEYWORD_ONLY, P.VAR_KEYWORD]
param_kind_to_int = dict(zip(PARAM_KINDS, range(len(PARAM_KINDS))))


def _find_insertion_index_by_kind(ps, kind):
    i = -1
    for i, p in enumerate(ps):
        if param_kind_to_int[p.kind] > param_kind_to_int[kind]:
            break
    else:
        i = i + 1
    return i


class SignatureEx(inspect.Signature):

    def drop_arg(self, argname, raise_if_not_found=False):
        ps = dict(self.parameters.items())
        if argname in ps:
            del ps[argname]
        elif raise_if_not_found:
            raise KeyError(f"'{argname}' not found in {list(ps.keys())}")
        return self.replace(parameters=ps.values())

    def add_arg(self, argname, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD, *, default=inspect.Parameter.empty, annotation=inspect.Parameter.empty):
        ps = list(self.parameters.values())
        ind = _find_insertion_index_by_kind(ps, kind)
        ps.insert(ind, inspect.Parameter(argname, kind,
                  default=default, annotation=annotation))
        return self.replace(parameters=ps)

    @property
    def has_return_annotation(self):
        return self.return_annotation is not self.empty

    def format_argument_signature(self):
        return _extract_argument_signature(str(self))

    def format_return_annotation(self):
        return _extract_return_annotation(str(self), self.has_return_annotation)

    def format_caller_argument_signature(self):
        return _create_caller_signature(self)
