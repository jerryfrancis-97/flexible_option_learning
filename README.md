# Simultaneous State-Dependent Updating of multi-options in Flexible Option Learning

### Abstract
Within intra-option learning, Flexible Option Learning was formulated to enable
simultaneous update of all option-policies which was consistent with their primitive
action choices. However, updating all options without any goal-related heuristic
reduces the degree of diversity of options within the option set, which is a major
drawback. We revisit and extend Flexible Option learning to introduce a state-
dependent multi-option update method to add more flexibility to the way multi-
option updates can be performed in the context of deep reinforcement learning.
Our method utilizes the concept of distance as a goal-related heuristic to generalize
the multi-option updates in different important transitions like bottleneck situations
in environment.


### Method
We follow all the notations and the assumptions for MDPs and options as mentioned in the Flexible
Option Learning Klissarov and Precup (2021). In this work, we introduce a state-dependent function
η(s) which is based on the concept of distance Distance from the current state s in an environment
till the goal G. Hence, η(s) can be written as follows,

η(s) = 1 / Distance(s, G)

where Distance can be Euclidean distance or the Manhattan (Cityblock) distance. Therefore,

Distance = Euclidean(X, Y) = sqrt[ (Xi − Xj)^2 + (Yi − Yj)^2 ] 

and

Distance = Cityblock(X, Y) = Abs(Xi − Xj) + Abs(Yi − Yj)

#### NB : Used for EEML'22 submission.
============================================================================================
## From Flexible Option Learning original git repo

This repository contains code for the paper [Flexible Option Learning](https://arxiv.org/abs/2112.03097) presented as a Spotlight at NeurIPS 2021. The implementation is based on [gym-miniworld](https://github.com/maximecb/gym-miniworld), OpenAI's  [baselines](https://github.com/openai/baselines) and the Option-Critic's [tabular implementation](https://github.com/jeanharb/option_critic/tree/master/fourrooms).


Contents:
- [FourRooms Experiments](#tabular-experiments-four-rooms)
- [Continuous Control Experiments](#continuous-control-mujoco)
- [Visual Navigation Experiments](#maze-navigation-miniworld)
- [Citation](#cite)





## Tabular Experiments (Four-Rooms)

#### Installation and Launch code

```
pip install gym==0.12.1
cd diagnostic_experiments/
python main_fixpol.py --multi_option # for experiments with fixed options
python main.py --multi_option # for experiments with learned options
```


## Continuous Control (MuJoCo)

#### Installation

```
virtualenv moc_cc --python=python3
source moc_cc/bin/activate
pip install tensorflow==1.12.0 
cd continuous_control
pip install -e . 
pip install gym==0.9.3
pip install mujoco-py==0.5.1
```
#### Launch

```
cd baselines/ppoc_int
python run_mujoco.py --switch --nointfc --env AntWalls --eta 0.9 --mainlr 8e-5 --intlr 8e-5 --piolr 8e-5
```


## Maze Navigation (MiniWorld)

#### Installation

```
virtualenv moc_vision --python=python3
source moc_vision/bin/activate
pip install tensorflow==1.13.1
cd vision_miniworld
pip install -e .
pip install gym==0.15.4
```

#### Launch

```
cd baselines/
# Run agent in first task
python run.py --alg=ppo2_options --env=MiniWorld-WallGap-v0 --num_timesteps 2500000 --save_interval 1000  --num_env 8 --noptions 4 --eta 0.7

# Load and run agent in transfer task
python run.py --alg=ppo2_options --env=MiniWorld-WallGapTransfer-v0 --load_path path/to/model --num_timesteps 2500000 --save_interval 1000  --num_env 8 --noptions 4 --eta 0.7
```


## Cite

If you find this work useful to you, please consider adding us to your references. 


```
@inproceedings{
klissarov2021flexible,
title={Flexible Option Learning},
author={Martin Klissarov and Doina Precup},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021},
url={https://openreview.net/forum?id=L5vbEVIePyb}
}
```
