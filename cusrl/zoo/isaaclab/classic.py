from cusrl.environment import make_isaaclab_env
from cusrl.preset import ppo
from cusrl.zoo.registry import register_experiment

__all__ = []

register_experiment(
    environment_name="Isaac-Ant-v0",
    algorithm_name="ppo",
    agent_factory_cls=ppo.AgentFactory,
    agent_factory_kwargs=dict(
        num_steps_per_update=32,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation_fn="ELU",
        lr=1e-3,
        sampler_epochs=5,
        sampler_mini_batches=4,
        orthogonal_init=False,
        entropy_loss_weight=0.0,
        desired_kl_divergence=0.015,
    ),
    training_env_factory=make_isaaclab_env,
    num_iterations=1000,
    save_interval=100,
)

register_experiment(
    environment_name="Isaac-Cartpole-v0",
    algorithm_name="ppo",
    agent_factory_cls=ppo.AgentFactory,
    agent_factory_kwargs=dict(
        num_steps_per_update=16,
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(32, 32),
        activation_fn="ELU",
        lr=1e-3,
        sampler_epochs=5,
        sampler_mini_batches=4,
        orthogonal_init=False,
        entropy_loss_weight=0.005,
        desired_kl_divergence=0.015,
    ),
    training_env_factory=make_isaaclab_env,
    num_iterations=150,
    save_interval=50,
)

register_experiment(
    environment_name="Isaac-Humanoid-v0",
    algorithm_name="ppo",
    agent_factory_cls=ppo.AgentFactory,
    agent_factory_kwargs=dict(
        num_steps_per_update=32,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation_fn="ELU",
        lr=1e-3,
        sampler_epochs=5,
        sampler_mini_batches=4,
        orthogonal_init=False,
        normalize_observation=True,
        entropy_loss_weight=0.0,
        desired_kl_divergence=0.012,
    ),
    training_env_factory=make_isaaclab_env,
    num_iterations=1000,
    save_interval=200,
)
