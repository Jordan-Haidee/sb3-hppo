Hybrid-Action-PPO

### How to use ###

set your action space belong to under lines

```
d = {}
d['continuous_action'] = spaces.Box(low = np.array([]), high=np.array([]), shape=(), dtype=np.float64)
d['discrete_action'] = spaces.Discrete()
self.action_space = spaces.Dict(d)
```

get action by step member function 

```
def step(self, action:np.ndarray):
    discrete_action = action[0]
    continuous_action action[1:]
```


### reference ###  
[Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space](https://arxiv.org/abs/1903.01344)
