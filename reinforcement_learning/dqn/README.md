# Deep Q Network

This is a NNabla implementation of DQN.

* [Mnih et.al., Human-level control through deep reinforcement learning.](https://www.nature.com/articles/nature14236)

##  Requirements

* gym
```
pip install gym
pip install gym[atari]
```

* TensorboardX
```
pip install tensorboardX
```

## Run

### Training

Run (see options by `-h`):

```
python train_atari.py
```

This will output the following files in a log folder (`.tmp.monitor` by default):

* Mean episodic rewards in each epoch over time.
* Learned models with `nnp` format.


### Playing learned model

Run (see options by `-h`):

```
python play_atari.py
```

### Training parameter difference between ours and OpenAI baseline 

This implementation uses [OpenAI baseline](https://github.com/openai/baselines)'s parameter by default.


|                             |  Ours | Baseline (defaults.py)  |
| ----                        |  ---- | ----                    |
| Learning Rate               | 1e-4  | 1e-4                    |
| Start Epsilon               | 1.0   | 1.0                     |
| Final Epsilon               | 0.01  | 0.01                    |
| Epsilon Decay Steps         | 1e6   | 1e6 (when max_step:1e8) |
| Dicount Factor (gamma)      | 0.99  | 0.99                    |
| Batch Size                  | 32    | 32                      |
| Replay Buffer Size          | 10000 | 10000                   |
| Target Network Update Freq. | 1000  | 1000                    |
| Learning Start Step         | 10000 | 10000                   |


## Atari Evaluation

There are 2 steps when evaluating Atari games.

1. Intermediate evaluation during training

    * In every 1M frames (250K steps), the mean reward is evaluated using the Q-Network parameter at that timestep. 
    * The evaluation step lasts for 500K frames (125K steps) but the last episode that exceeeds 125K timesteps is not used for evaluation.
    * epsilon is set to 0.05 (not greedy).

2. Final evaluation for reporting

    * The Q-Network parameter (.nnp file) with the best mean reward is used. You can just look through `Eval-Score.series.txt` and find the maximum scored steps. Note that `qnet_XXX.nnp`'s `XXX` represents **steps** and not **frames**, whereas the score output in `Eval-Score.series.txt` are **frames**.

    * Using the best .nnp file, by running as shown below, the specified game is played for 30 episodes with a termination threshold of 4500 steps.

        Run (see options by `-h`):
        ```
        python eval_atari.py
        ```


### Our Evaluation

Below we show the evaluation results according to the evaluation method above. Though we trained less (10M steps) than the [original nature paper](https://www.nature.com/articles/nature14236) (50M steps), due to efficient parameters by OpenAI baseline, most of the games scored better on ours. NNabla DQN won 34 of the games, Nature DQN won 14 of the games, and 1 game was tied.

| Game | NNabla DQN (10M steps training) | Nature DQN (50M steps training) |
| ---- | ---- | ---- |
|AirRaid|6313.3|N/A|
|Alien|1836.7|3069|
|Amidar|530.5|739.5|
|Assault|5435.8|3359|
|Asterix|11146.7|6012|
|Asteroids|1166.0|1629|
|Atlantis|412240.0|85641|
|BankHeist|1005.3|429.7|
|BattleZone|21966.7|26300|
|BeamRider|7216.5|6846|
|Berzerk|690.7|N/A|
|Bowling|58.1|42.4|
|Boxing|99.1|71.8|
|Breakout|305.1|401.2|
|Carnival|5781.7|N/A|
|Centipede|2118.7|8309|
|ChopperCommand|1160.0|6687|
|CrazyClimber|94890.0|114103|
|DemonAttack|18954.2|9711|
|DoubleDunk|-2.5|-18.1|
|Enduro|1270.7|301.8|
|FishingDerby|25.7|-0.8|
|Freeway|33.6|30.3|
|Frostbite|2266.3|328.3|
|Gopher|9895.3|8520|
|Gravitar|598.3|306.7|
|Hero|20741.3|19950|
|IceHockey|-0.7|-1.6|
|Jamesbond|835.0|576.7|
|JourneyEscape|-1166.7|N/A|
|Kangaroo|14940.0|6740|
|Krull|8648.7|3805|
|KungFuMaster|36623.3|23270|
|MontezumaRevenge|0.0|0.0|
|MsPacman|2530.7|2311|
|NameThisGame|6071.0|7257|
|Phoenix|6848.0|N/A|
|Pitfall|0.0|N/A|
|Pong|21.0|18.9|
|Pooyan|2964.7|N/A|
|PrivateEye|-40.0|1788|
|Qbert|15020.0|10596|
|Riverraid|10703.3|8316|
|RoadRunner|54756.7|18257|
|Robotank|59.7|51.6|
|Seaquest|3417.3|5286|
|Skiing|-9311.8|N/A|
|Solaris|2.7|N/A|
|SpaceInvaders|1177.5|1976|
|StarGunner|39823.3|57997|
|Tennis|-1.0|-2.5|
|TimePilot|7116.7|5947|
|Tutankham|244.6|186.7|
|UpNDown|22355.3|8456|
|Venture|1083.3|380.0|
|VideoPinball|311023.5|42684|
|WizardOfWor|1520.0|3393|
|YarsRevenge|28096.1|N/A|
|Zaxxon|5336.7|4977|
