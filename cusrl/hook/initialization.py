import itertools
import math
from typing import Literal

from torch import nn

from cusrl.template import ActorCritic, Hook

__all__ = ["ModuleInitialization"]


class ModuleInitialization(Hook[ActorCritic]):
    """A hook to initialize the weights of the actor and critic networks.

    This hook applies orthogonal initialization to linear, recurrent, and multi-head
    attention layers, and Kaiming normal initialization to convolutional layers.
    This is a common practice in reinforcement learning to improve training stability.

    Args:
        scale (float, optional):
            The gain factor for orthogonal initialization. Defaults to `sqrt(2)`.
        scale_dist (float, optional):
            The gain factor for the final linear layer of the actor's distribution head.
            A smaller value is often used to ensure small initial action outputs.
            Defaults to `sqrt(2) * 0.1`.
        zero_bias (bool, optional):
            If True, all bias terms in the initialized layers will be set to zero.
            Defaults to True.
        conv_a (float, optional):
            The 'a' parameter (negative slope) for Kaiming initialization, used with
            leaky_relu. Defaults to 0.0.
        conv_mode (Literal["fan_in", "fan_out"], optional):
            The mode for Kaiming initialization. Defaults to "fan_in".
        conv_nonlinearity (Literal["relu", "leaky_relu"], optional):
            The nonlinearity to use for calculating the Kaiming gain. Defaults to
            "leaky_relu".
        init_actor (bool, optional):
            Whether to initialize the actor network. Defaults to True.
        init_critic (bool, optional):
            Whether to initialize the critic network. Defaults to True.
        distribution_std (float | None, optional):
            If provided, sets a fixed standard deviation for the action distribution.
            Defaults to None.
    """

    def __init__(
        self,
        scale: float = math.sqrt(2),
        scale_dist: float = math.sqrt(2) * 0.1,
        zero_bias: bool = True,
        conv_a: float = 0.0,
        conv_mode: Literal["fan_in", "fan_out"] = "fan_in",
        conv_nonlinearity: Literal["relu", "leaky_relu"] = "leaky_relu",
        init_actor: bool = True,
        init_critic: bool = True,
        distribution_std: float | None = None,
    ):
        self.scale = scale
        self.scale_dist = scale_dist
        self.zero_bias = zero_bias
        self.conv_a = conv_a
        self.conv_mode = conv_mode
        self.conv_nonlinearity = conv_nonlinearity
        self.init_actor = init_actor
        self.init_critic = init_critic
        self.distribution_std = distribution_std

    def init(self):
        if self.init_actor:
            for module in itertools.chain(self.agent.actor.modules()):
                self._init_module(module, self.scale, self.zero_bias)
            if self.scale_dist != self.scale:
                self._init_linear(self.agent.actor.distribution.mean_head, self.scale_dist, self.zero_bias)
        if self.distribution_std is not None:
            self.agent.actor.set_distribution_std(self.distribution_std)
        if self.init_critic:
            for module in itertools.chain(self.agent.critic.modules()):
                self._init_module(module, self.scale, self.zero_bias)

    def _init_module(self, module: nn.Module, scale: float, zero_bias: bool):
        if isinstance(module, nn.Linear):
            self._init_linear(module, scale, zero_bias)
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            self._init_gru_lstm(module, scale, zero_bias)
        elif isinstance(module, nn.MultiheadAttention):
            self._init_mha(module, scale, zero_bias)
        elif isinstance(module, nn.Conv2d):
            self._init_conv2d(module, zero_bias)

    def _init_linear(self, module: nn.Linear, scale: float, zero_bias: bool):
        nn.init.orthogonal_(module.weight, gain=scale)
        if zero_bias and module.bias is not None:
            nn.init.zeros_(module.bias)

    def _init_gru_lstm(self, module: nn.GRU | nn.LSTM, scale: float, zero_bias: bool):
        for i in range(module.num_layers):
            nn.init.orthogonal_(getattr(module, f"weight_hh_l{i}"), gain=scale)
            nn.init.orthogonal_(getattr(module, f"weight_ih_l{i}"), gain=scale)
            if zero_bias and getattr(module, f"bias_hh_l{i}") is not None:
                nn.init.zeros_(getattr(module, f"bias_hh_l{i}"))
                nn.init.zeros_(getattr(module, f"bias_ih_l{i}"))

    def _init_mha(self, module: nn.MultiheadAttention, scale: float, zero_bias: bool):
        if module.in_proj_weight is not None:
            nn.init.orthogonal_(module.in_proj_weight, gain=scale)
        else:
            nn.init.orthogonal_(module.q_proj_weight, gain=scale)
            nn.init.orthogonal_(module.k_proj_weight, gain=scale)
            nn.init.orthogonal_(module.v_proj_weight, gain=scale)

        if zero_bias:
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            if module.bias_k is not None:
                nn.init.zeros_(module.bias_k)
            if module.bias_v is not None:
                nn.init.zeros_(module.bias_v)

    def _init_conv2d(self, module: nn.Conv2d, zero_bias: bool):
        nn.init.kaiming_normal_(
            module.weight,
            a=self.conv_a,
            mode=self.conv_mode,
            nonlinearity=self.conv_nonlinearity,
        )
        if zero_bias and module.bias is not None:
            nn.init.zeros_(module.bias)
