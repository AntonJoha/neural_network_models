import unittest

from RL.ddpg_agent import DDPG, DDPGAgent


class TestDDPGAgentImport(unittest.TestCase):
    def test_ddpg_agent_aliases_ddpg(self):
        self.assertIs(DDPGAgent, DDPG)


if __name__ == "__main__":
    unittest.main()
