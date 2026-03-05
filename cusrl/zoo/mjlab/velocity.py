from cusrl.environment import make_mjlab_env
from cusrl.environment.mjlab import MjlabPlayer
from cusrl.preset import ppo
from cusrl.zoo.registry import register_experiment

register_experiment(
    environment_name=[
        "Mjlab-Velocity-Flat-Unitree-G1",
        "Mjlab-Velocity-Flat-Unitree-Go1",
        "Mjlab-Velocity-Rough-Unitree-G1",
        "Mjlab-Velocity-Rough-Unitree-Go1",
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
        entropy_loss_weight=0.01,
        desired_kl_divergence=0.015,
    ),
    training_env_factory=make_mjlab_env,
    player_class=MjlabPlayer,
    playing_env_kwargs={"play": True},
    num_iterations=20000,
    save_interval=500,
)
