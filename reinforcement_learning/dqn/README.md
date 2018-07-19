# Deep Q Network

This is a NNabla implementation of DQN.

* [Mnih et.al., Human-level control through deep reinforcement learning.](https://www.nature.com/articles/nature14236)

##  Requirements

* gym
```
pip install gym
```

## Run

### Training

Run (see options by `-h`):

```
python train_classific_control.py
```

This will outputs the following files in a log folder (`.tmp.monitor` by default):

* Mean episodic rewards in each epoch over time.
* Learned models with `nnp` format.


### Playing learned model

Run (see options by `-h`):

```
python play_classic_control.py
```
