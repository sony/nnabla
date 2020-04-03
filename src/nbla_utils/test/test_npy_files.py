import pytest
import numpy as np
import zlib
import csv
import os
from collections import OrderedDict


DATA_NUM = 640
RANGE = 80
CACHE_FOLDER = "cache_npy"


def save_variable_list(variables):
    names = [k for k, _ in variables.items()]
    with open(os.path.join(CACHE_FOLDER, "cache_info.csv"), "w") as f:
        for line in names:
            f.write(line + "\n")


def save_file_list(cache_file_names):
    with open(os.path.join(CACHE_FOLDER, "cache_index.csv"), "w") as f:
        wr = csv.writer(f)
        wr.writerows(cache_file_names)


def save_variable_to_cache_file(variables):
    num_of_files = DATA_NUM // RANGE
    if num_of_files * RANGE < DATA_NUM:
        num_of_files += 1

    split_variables = OrderedDict()
    for k, v in variables.items():
        split_variables[k] = np.split(v, num_of_files, axis=0)

    cache_file_names = []
    range_start = 0
    range_end = range_start + RANGE - 1
    for i in range(num_of_files):
        cache_file_name = os.path.join(
            CACHE_FOLDER, "cache_sliced_{:08d}_{:08d}.npy".format(range_start, range_end))
        range_start = range_end
        range_end += RANGE
        cache_file_names.append([os.path.basename(cache_file_name), RANGE])

        with open(cache_file_name, "wb") as f:
            for k, v in split_variables.items():
                np.save(f, v[i])

    save_file_list(cache_file_names)
    save_variable_list(variables)


@pytest.fixture(scope="module", autouse=True)
def prepare_environment_fixture():
    if not os.path.exists(CACHE_FOLDER):
        os.makedirs(CACHE_FOLDER)
    else:
        n = 0
        while os.path.exists(CACHE_FOLDER + str(n)):
            n += 1
    print("{} is checked.".format(CACHE_FOLDER))


@pytest.mark.parametrize('var_names', [["x0", "x1", "x2"]])
@pytest.mark.parametrize('dims', [3])
def test_generate_self_verf_data(var_names, dims):
    variables = OrderedDict()
    k = 0
    y = np.zeros([DATA_NUM, len(var_names)]).astype(np.uint32)
    for n in var_names:
        shape = list(np.random.randint(4, 28, size=(dims,)))
        variables[n] = np.random.randint(
            127, 255, size=[DATA_NUM] + shape).astype(np.uint32)
        for i in range(variables[n].shape[0]):
            idx = tuple([i] + [0] * dims)
            variables[n][idx] = i
            y[i, k] = zlib.crc32(variables[n][i, ...].tobytes('C'))
        k += 1
        variables['y'] = y
    save_variable_to_cache_file(variables)
