from cusrl.environment import make_isaaclab_env
from cusrl.preset import ppo
from cusrl.zoo.registry import register_experiment

__all__ = []

register_experiment(
    environment_name=[
        "Isaac-Velocity-Flat-Anymal-B-v0",
        "Isaac-Velocity-Flat-Anymal-C-v0",
        "Isaac-Velocity-Flat-Anymal-D-v0",
        "Isaac-Velocity-Flat-Unitree-A1-v0",
        "Isaac-Velocity-Flat-Unitree-Go1-v0",
        "Isaac-Velocity-Flat-Unitree-Go2-v0",
        "Isaac-Velocity-Flat-Spot-v0",
    ],
    algorithm_name="ppo",
    agent_factory_cls=ppo.AgentFactory,
    agent_factory_kwargs=dict(
        num_steps_per_update=24,
        actor_hidden_dims=(128, 128, 128),
        critic_hidden_dims=(128, 128, 128),
        activation_fn="ELU",
        lr=1e-3,
        sampler_epochs=5,
        sampler_mini_batches=4,
        orthogonal_init=False,
        entropy_loss_weight=0.005,
        desired_kl_divergence=0.015,
    ),
    training_env_factory=make_isaaclab_env,
    playing_env_kwargs={"play": True},
    num_iterations=300,
    save_interval=100,
)

register_experiment(
    environment_name=[
        "Isaac-Velocity-Rough-Anymal-B-v0",
        "Isaac-Velocity-Rough-Anymal-C-v0",
        "Isaac-Velocity-Rough-Anymal-D-v0",
        "Isaac-Velocity-Rough-Unitree-A1-v0",
        "Isaac-Velocity-Rough-Unitree-Go1-v0",
        "Isaac-Velocity-Rough-Unitree-Go2-v0",
    ],
    algorithm_name="ppo",
    agent_factory_cls=ppo.AgentFactory,
    agent_factory_kwargs=dict(
        num_steps_per_update=24,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation_fn="ELU",
        lr=1e-3,
        sampler_epochs=5,
        sampler_mini_batches=4,
        orthogonal_init=False,
        entropy_loss_weight=0.005,
        desired_kl_divergence=0.015,
    ),
    training_env_factory=make_isaaclab_env,
    playing_env_kwargs={"play": True},
    num_iterations=1500,
    save_interval=100,
)
