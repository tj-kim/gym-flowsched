import sys
import multiprocessing
import os.path as osp
import gym
import gym_flowsched
from collections import defaultdict
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module

from baselines.common.vec_env.vec_normalize import VecNormalize

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, extra_args):
    env_type, env_id = get_env_type(args.env)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.Logger.CURRENT.dir, "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args.env)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
       config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
       config.gpu_options.allow_growth = True
       get_session(config=config)

       flatten_dict_observations = alg not in {'her'}
       env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

       if env_type == 'mujoco':
           env = VecNormalize(env)

    return env


def get_env_type(env_id):
    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}



def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)
    num_episodes = 1000
    flowtime_episodes = np.zeros(num_episodes)
    nF, nL = 10, 6
    flow_time_link = np.zeros((nL, nF))
    cum_flowtime = np.zeros((num_episodes,))
    rew = 0

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure()
    else:
        logger.configure(format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()
   
    model, env = train(args, extra_args)


    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)
        print(dir(model))

    if args.play:
        logger.log("Running trained model")
        #env = build_env(args)
        

        obs = env.reset()
        print(obs)

        # If wanting to use different parameters of the model, 
        # link index iL can be included in state as an input
        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        i_episode = 0
        actions = []
        while True:
            for iL in range(nL):
                # The model.step() takes the observation on link iL as input
                # which means the corresponding outputted action_iL is the protocol
                # selected only for link iL
                if state is not None:
                    action_iL, _, state, _ = model.step(obs[iL], S=state, M=dones)
                else:
                    action_iL, _, _, _ = model.step(obs[iL])
                actions.append(action_iL)
            # actions for all links are inputted into the env.step()
            # obs, rew, and actions should all be revised to be nL-by-1 at flowsched_multi_path_env.py
            obs, rew, done, _ = env.step(actions)
            done = done.any() if isinstance(done, np.ndarray) else done
            if done:
                flowtime_episodes[i_episode] = env.render()
                i_episode += 1
                if i_episode >= num_episodes:
                    break
                obs = env.reset()

    env.close()

    # Write data into file
    cum_flowtime = np.cumsum(flowtime_episodes)
    np.savetxt('data_mp_normal.txt', cum_flowtime) # Used to plot multipath performance under synthetic data
    # np.savetxt('data_mp_normal_no_training.txt', cum_flowtime) # Used to plot a2c without pre-training under synthetic data
    # np.savetxt('data_testbed_a2c.txt', cum_flowtime) # Used to plot multipath performance under BAE testbed data

    return model

if __name__ == '__main__':
    main(sys.argv)