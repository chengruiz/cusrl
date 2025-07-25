from functools import partial

import cusrl
from cusrl.environment import make_isaaclab_env
from cusrl.zoo.registry import register_experiment

__all__ = []


class AgentFactory(cusrl.preset.amp.AgentFactory):
    def from_environment(self, environment: cusrl.environment.IsaacLabEnvAdapter) -> cusrl.ActorCritic:
        self.get_hook("AdversarialMotionPrior").dataset_source = partial(
            environment.wrapped.collect_reference_motions, 200000
        )
        return super().from_environment(environment)


register_experiment(
    environment_name="Isaac-Humanoid-AMP-Dance-Direct-v0",
    algorithm_name="ppo",
    agent_factory_cls=AgentFactory,
    agent_factory_kwargs=dict(
        num_steps_per_update=16,
        actor_hidden_dims=(1024, 512),
        critic_hidden_dims=(1024, 512),
        normalize_observation=True,
        activation_fn="ReLU",
        lr=5e-5,
        sampler_epochs=6,
        sampler_mini_batches=2,
        orthogonal_init=False,
        init_distribution_std=0.05,
        extrinsic_reward_scale=0.0,
        amp_discriminator_hidden_dims=(1024, 512),
        entropy_loss_weight=0.001,
    ),
    training_env_factory=make_isaaclab_env,
    num_iterations=3000,
    save_interval=500,
)

register_experiment(
    environment_name="Isaac-Humanoid-AMP-Run-Direct-v0",
    algorithm_name="ppo",
    agent_factory_cls=AgentFactory,
    agent_factory_kwargs=dict(
        num_steps_per_update=16,
        actor_hidden_dims=(1024, 512),
        critic_hidden_dims=(1024, 512),
        normalize_observation=True,
        activation_fn="ReLU",
        lr=5e-5,
        sampler_epochs=6,
        sampler_mini_batches=2,
        orthogonal_init=False,
        init_distribution_std=0.05,
        extrinsic_reward_scale=0.0,
        amp_discriminator_hidden_dims=(1024, 512),
        entropy_loss_weight=0.001,
    ),
    training_env_factory=make_isaaclab_env,
    num_iterations=3000,
    save_interval=500,
)

register_experiment(
    environment_name="Isaac-Humanoid-AMP-Walk-Direct-v0",
    algorithm_name="ppo",
    agent_factory_cls=AgentFactory,
    agent_factory_kwargs=dict(
        num_steps_per_update=16,
        actor_hidden_dims=(1024, 512),
        critic_hidden_dims=(1024, 512),
        normalize_observation=True,
        activation_fn="ReLU",
        lr=5e-5,
        sampler_epochs=6,
        sampler_mini_batches=2,
        orthogonal_init=False,
        init_distribution_std=0.05,
        extrinsic_reward_scale=0.0,
        amp_discriminator_hidden_dims=(1024, 512),
        entropy_loss_weight=0.001,
    ),
    training_env_factory=make_isaaclab_env,
    num_iterations=3000,
    save_interval=500,
)
