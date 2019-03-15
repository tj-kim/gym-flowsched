import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='FlowSched-v0',
    entry_point='gym_flowsched.envs:FlowSchedEnv',
    )