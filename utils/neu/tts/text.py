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

import re
import unicodedata

import inflect

engine = inflect.engine()


def text_normalize(text, vocab):
    """Normalize an input text.

    Args:
        text (str): An input text.
        vocab (str): A string containing alphabets.

    Returns:
        str: A text containing only given alphabets.
    """
    # remove accents
    text = ''.join(ch for ch in unicodedata.normalize('NFD', text)
                   if unicodedata.category(ch) != 'Mn')
    text = re.sub(
        r"(\d+)",
        lambda x: engine.number_to_words(x.group(0)).replace(',', ''),
        text
    )
    text = re.sub("[^{}]".format(vocab), " ", text)
    text = re.sub("[ ]+", " ", text)

    return text
