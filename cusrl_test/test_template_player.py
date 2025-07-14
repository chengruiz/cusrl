import cusrl
from cusrl.template.player import Player
from cusrl_test import create_dummy_env


def test_player():
    environment = create_dummy_env()
    agent = cusrl.preset.ppo.AgentFactory()
    Player(environment, agent, num_steps=100).run_playing_loop()
    Player(environment, agent, num_steps=100, timestep=0.01).run_playing_loop()
