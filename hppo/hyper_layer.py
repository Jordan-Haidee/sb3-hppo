from typing import Dict, List, Type, Union, Tuple
import torch as th
from torch import nn
from stable_baselines3.common.utils import get_device


class HyMlpExtractor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        policy_net_con: List[nn.Module] = []
        policy_net_disc: List[nn.Module] = []
        value_net: List[nn.Module] = []
        last_layer_dim_pi = feature_dim
        last_layer_dim_vf = feature_dim

        if isinstance(net_arch, dict):
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network

        else:
            pi_layers_dims = vf_layers_dims = net_arch
        for curr_layer_dim in pi_layers_dims:
            policy_net_con.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net_con.append(activation_fn())
            policy_net_disc.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net_disc.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.policy_net_con = nn.Sequential(*policy_net_con).to(device)
        self.policy_net_disc = nn.Sequential(*policy_net_disc).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return (
            self.forward_actor_disc(features),
            self.forward_actor_con(features),
            self.forward_critic(features),
        )

    def forward_actor_disc(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net_disc(features)

    def forward_actor_con(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net_con(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)
