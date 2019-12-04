# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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

import nnabla as nn
from nnabla.logger import logger
import os
import json


def save_checkpoint(path, current_iter, solvers):
    """Saves the checkpoint file which contains the params and its state info.

        Args:
            path: Path to the directory the checkpoint file is stored in.
            current_iter: Current iteretion of the training loop.
            solvers: A dictionary about solver's info, which is like;
                     solvers = {"identifier_for_solver_0": solver_0,
                               {"identifier_for_solver_1": solver_1, ...}
                     The keys are used just for state's filenames, so can be anything.
                     Also, you can give a solver object if only one solver exists.
                     Then, the "" is used as an identifier.

        Examples:
            # Create computation graph with parameters.
            pred = construct_pred_net(input_Variable, ...)

            # Create solver and set parameters.
            solver = S.Adam(learning_rate)
            solver.set_parameters(nn.get_parameters())

            # If you have another_solver like,
            # another_solver = S.Sgd(learning_rate)
            # another_solver.set_parameters(nn.get_parameters())

            # Training loop.
            for i in range(start_point, max_iter):
                pred.forward()
                pred.backward()
                solver.zero_grad()
                solver.update()
                save_checkpoint(path, i, solver)

                # If you have another_solver,
                # save_checkpoint(path, i,
                      {"solver": solver, "another_solver": another})

        Notes:
            It generates the checkpoint file (.json) which is like;
            checkpoint_1000 = {
                    "":{
                        "states_path": <path to the states file>
                        "params_names":["conv1/conv/W", ...],
                        "num_update":1000
                       },
                    "current_iter": 1000
                    }

            If you have multiple solvers.
            checkpoint_1000 = {
                    "generator":{
                        "states_path": <path to the states file>,
                        "params_names":["deconv1/conv/W", ...],
                        "num_update":1000
                       },
                    "discriminator":{
                        "states_path": <path to the states file>,
                        "params_names":["conv1/conv/W", ...],
                        "num_update":1000
                       },
                    "current_iter": 1000
                    }

    """

    if isinstance(solvers, nn.solver.Solver):
        solvers = {"": solvers}

    checkpoint_info = dict()

    for solvername, solver_obj in solvers.items():
        prefix = "{}_".format(solvername.replace(
            "/", "_")) if solvername else ""
        partial_info = dict()

        # save solver states.
        states_fname = prefix + 'states_{}.h5'.format(current_iter)
        states_path = os.path.join(path, states_fname)
        solver_obj.save_states(states_path)
        partial_info["states_path"] = states_path

        # save registered parameters' name. (just in case)
        params_names = [k for k in solver_obj.get_parameters().keys()]
        partial_info["params_names"] = params_names

        # save the number of solver update.
        num_update = getattr(solver_obj.get_states()[params_names[0]], "t")
        partial_info["num_update"] = num_update

        checkpoint_info[solvername] = partial_info

    # save parameters.
    params_fname = 'params_{}.h5'.format(current_iter)
    params_path = os.path.join(path, params_fname)
    nn.parameter.save_parameters(params_path)
    checkpoint_info["params_path"] = params_path
    checkpoint_info["current_iter"] = current_iter

    checkpoint_fname = 'checkpoint_{}.json'.format(current_iter)
    filename = os.path.join(path, checkpoint_fname)

    with open(filename, 'w') as f:
        json.dump(checkpoint_info, f)

    logger.info("Checkpoint save (.json): {}".format(filename))

    return


def load_checkpoint(path, solvers):
    """Given the checkpoint file, loads the parameters and solver states.

        Args:
            path: Path to the checkpoint file.
            solvers: A dictionary about solver's info, which is like;
                     solvers = {"identifier_for_solver_0": solver_0,
                               {"identifier_for_solver_1": solver_1, ...}
                     The keys are used for retrieving proper info from the checkpoint.
                     so must be the same as the one used when saved.
                     Also, you can give a solver object if only one solver exists.
                     Then, the "" is used as an identifier.

        Returns:
            current_iter: The number of iteretions that the training resumes from.
                          Note that this assumes that the numbers of the update for
                          each solvers is the same.

        Examples:
            # Create computation graph with parameters.
            pred = construct_pred_net(input_Variable, ...)

            # Create solver and set parameters.
            solver = S.Adam(learning_rate)
            solver.set_parameters(nn.get_parameters())

            # AFTER setting parameters.
            start_point = load_checkpoint(path, solver)

            # Training loop.

        Notes:
            It requires the checkpoint file. For details, refer to save_checkpoint;
            checkpoint_1000 = {
                    "":{
                        "states_path": <path to the states file>
                        "params_names":["conv1/conv/W", ...],
                        "num_update":1000
                       },
                    "current_iter": 1000
                    }

            If you have multiple solvers.
            checkpoint_1000 = {
                    "generator":{
                        "states_path": <path to the states file>,
                        "params_names":["deconv1/conv/W", ...],
                        "num_update":1000
                       },
                    "discriminator":{
                        "states_path": <path to the states file>,
                        "params_names":["conv1/conv/W", ...],
                        "num_update":1000
                       },
                    "current_iter": 1000
                    }

    """

    assert os.path.isfile(path), "checkpoint file not found"

    with open(path, 'r') as f:
        checkpoint_info = json.load(f)

    if isinstance(solvers, nn.solver.Solver):
        solvers = {"": solvers}

    logger.info("Checkpoint load (.json): {}".format(path))

    # load parameters (stored in global).
    params_path = checkpoint_info["params_path"]
    assert os.path.isfile(params_path), "parameters file not found."

    nn.parameter.load_parameters(params_path)

    for solvername, solver_obj in solvers.items():
        partial_info = checkpoint_info[solvername]
        if set(solver_obj.get_parameters().keys()) != set(partial_info["params_names"]):
            logger.warning("Detected parameters do not match.")

        # load solver states.
        states_path = partial_info["states_path"]
        assert os.path.isfile(states_path), "states file not found."

        # set solver states.
        solver_obj.load_states(states_path)

    # get current iteration. note that this might differ from the numbers of update.
    current_iter = checkpoint_info["current_iter"]

    return current_iter
