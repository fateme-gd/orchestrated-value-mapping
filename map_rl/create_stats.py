from __future__ import absolute_import

from absl import app
from absl import flags
from absl import logging

from dopamine.discrete_domains.run_experiment import load_gin_configs


import numpy as np
from map_rl import map_dqn_agent
from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import checkpointer
import gin
import tensorflow as tf


flags.DEFINE_string('base_dir', None,
                        'Base directory to host all required sub-directories.')
    
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files (e.g.'
    '"dopamine/agents/dqn/dqn.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_environment.game_name="Pong"").')


FLAGS = flags.FLAGS

def create_environment(game_name='MsPacman'):
    environment = atari_lib.create_atari_environment()
    return environment

@gin.configurable
class RandomDQNAgent(map_dqn_agent.MapDQNAgent):
    # def __init__(self, sess, num_actions):
    #     super(RandomDQNAgent, self).__init__(sess, num_actions)
        
    def step(self, reward, observation):
        """Returns a random action."""
        return np.random.randint(self.num_actions)
    
    # def load_checkpoint(self):
    #     """Load model from checkpoint."""
    #     saver = tf.train.Saver()
    #     saver.restore(self.sess, self.checkpoint_path)


# Main function to create the environment, agent, and run the agent
def main(unused_argv):
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.disable_v2_behavior()
    load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)

    checkpoint_path = FLAGS.base_dir  # Set your checkpoint path here
    checkpoint_file_prefix = 'ckpt'

    
    env = create_environment()
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    sess = tf.compat.v1.Session('', config=config)
    agent = RandomDQNAgent(sess, num_actions=env.action_space.n)

    sess.run(tf.compat.v1.global_variables_initializer())


    _checkpointer = checkpointer.Checkpointer(checkpoint_path,
                                                   checkpoint_file_prefix)
    start_iteration = 0 
    latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(checkpoint_path)
    print("path: ", checkpoint_path)
    print("latest: " , latest_checkpoint_version)
    if latest_checkpoint_version >= 0:
      experiment_data = _checkpointer.load_checkpoint(latest_checkpoint_version)
    agent.unbundle(checkpoint_path, latest_checkpoint_version, experiment_data)
    
    observation = env.reset()
    reward = 0
    done = False
    
    # Run the agent in the environment for one episode
    while not done:
        action = agent.step(reward, observation)
        observation, reward, done, _ = env.step(action)
        print(f"Action: {action}, Reward: {reward}")



if __name__ == '__main__':
    flags.mark_flag_as_required('base_dir')
    app.run(main)