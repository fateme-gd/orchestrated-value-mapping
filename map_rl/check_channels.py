

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf

import gin


from map_rl import models

MapDQNNetworkType = collections.namedtuple('map_dqn_network', 
    ['q_values', 'q_values_on_heads', 'q_tilde_values_on_heads'])

@gin.configurable    
class modifiedNetwork(models.MapDQNNetwork):
    def call(self, state):
        x = tf.cast(state, tf.float32)
        x = x / 255
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        if not self.use_nonlinear_heads:
            x = self.dense_torso(x)
        if self.use_gradscaling:
            N = len(self.lambdas_on_heads)
            gradscale_mult = 1.0 / np.sqrt(N)
            t = tf.multiply(x, gradscale_mult)
            x = t + tf.stop_gradient(x - t)
        x = tf.cast(x, self.tf_float)

        q_tilde_values_on_heads = []
        q_values_on_heads = []
        # Form of 'map_func_params', e.g., for the polar reward decomposition: 
        # [[pos_Delta, c], [neg_Delta, c]]. Each internal list is for one head.
        for hid, (head_dense, head_map_params, head_inverse_map_func) in enumerate(zip(self.dense_heads, self.map_func_params, self.inverse_map_funcs)):
            if self.use_nonlinear_heads:
                q_tilde_values = self.dense_heads_torso[hid](x)
                q_tilde_values = head_dense(q_tilde_values)
            else:
                q_tilde_values = head_dense(x)
            q_tilde_values_on_heads.append(q_tilde_values)
            # Inverse mapping.
            q_values_on_heads.append(head_inverse_map_func(q_tilde_values, *head_map_params))
    
        # Aggregate Q-values across heads. 
        first_head = True
        for head_lambda, head_q_values in zip(self.lambdas_on_heads, q_values_on_heads):
            if first_head:
                q_values = head_lambda * head_q_values
                first_head = False
            else:
                q_values += head_lambda * head_q_values

        return MapDQNNetworkType(q_values, q_values_on_heads, q_tilde_values_on_heads)

# To Do use the normal train.py but since update the gin config in run time (or change the config)