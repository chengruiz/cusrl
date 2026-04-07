from cusrl.environment import make_isaaclab_env
from cusrl.preset import PpoAgentFactory
from cusrl.zoo.registry import register_experiment

register_experiment(
    environment_name=[
        "RobotLab-Isaac-Velocity-Rough-Anymal-D-v0",
        "RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0",
        "RobotLab-Isaac-Velocity-Rough-HandStand-Unitree-A1-v0",
        "RobotLab-Isaac-Velocity-Rough-Unitree-B2-v0",
        "RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0",
    ],
    algorithm_name="ppo",
    agent_meta_factory=PpoAgentFactory,
    agent_meta_factory_kwargs=dict(
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
    training_env_factory=make_isaaclab_env,
    training_env_factory_kwargs={"extensions": ["robot_lab"]},
    num_iterations=20000,
    checkpoint_interval=500,
)
