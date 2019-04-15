import numpy as np
import os


CACHE_FOLDER = "cache_npy"


def del_file_from_cache_folder(cache_folder):
    for f in os.listdir(cache_folder):
        path = os.path.join(cache_folder, f)
        try:
            if os.path.isfile(path):
                os.unlink(path)
        except Exception as e:
            print(e)


def gen_npy_file(name, v):
    v = v.astype(np.float32)
    np.save(os.path.join(CACHE_FOLDER, name + ".npy"), v)


def gen_variables(vars, batch_size):
    for k, v in vars.items():
        shape = [batch_size] + v
        variable = np.random.random(shape)
        gen_npy_file(k, variable)


variables = {
  "input_x": [3, 16, 16],
  "input_y": [6, 16],
  "input_z": [8]
}

del_file_from_cache_folder(CACHE_FOLDER)
gen_variables(variables, 10)
