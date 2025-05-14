import gymnasium as gym


class SB3HyPPOWrapper(gym.Wrapper):
    """
    Wrapper for the HyPPO algorithm in Stable Baselines3.
    """

    def __init__(self, env):
        super().__init__(env)

        d = {}
        d["discrete_action"] = env.action_space[0]
        d["continuous_action"] = env.action_space[1]
        self.action_space = gym.spaces.Dict(d)
        self.env = env

    def step(self, action):
        _a = (int(action[0]), action[1:])
        state, reward, terminated, truncated, info = self.env.step(_a)
        return state, reward, terminated, truncated, info

    def render(self):
        return self.env.render()
