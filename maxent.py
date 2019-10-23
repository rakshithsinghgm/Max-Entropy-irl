import numpy as np
import numpy.random as rand
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


def build_trans_mat_gridworld():
  # 5x5 gridworld laid out like:
  # 0  1  2  3  4
  # 5  6  7  8  9
  # ...
  # 20 21 22 23 24
  # where 24 is a goal state that always transitions to a
  # special zero-reward terminal state (25) with no available actions
  trans_mat = np.zeros((26,4,26))

  # NOTE: the following iterations only happen for states 0-23.
  # This means terminal state 25 has zero probability to transition to any state,
  # even itself, making it terminal, and state 24 is handled specially below.

  # Action 0 = down
  for s in range(24):
    if s < 20:
      trans_mat[s,0,s+5] = 1
    else:
      trans_mat[s,0,s] = 1

  # Action 1 = up
  for s in range(24):
    if s >= 5:
      trans_mat[s,1,s-5] = 1
    else:
      trans_mat[s,1,s] = 1

  # Action 2 = left
  for s in range(24):
    if s%5 > 0:
      trans_mat[s,2,s-1] = 1
    else:
      trans_mat[s,2,s] = 1

  # Action 3 = right
  for s in range(24):
    if s%5 < 4:
      trans_mat[s,3,s+1] = 1
    else:
      trans_mat[s,3,s] = 1

  # Finally, goal state always goes to zero reward terminal state
  for a in range(4):
    trans_mat[24,a,25] = 1

  return trans_mat

def calcMaxEntPolicy(trans_mat, horizon, r_weights, state_features):
  """
  For a given reward function and horizon, calculate the MaxEnt policy that gives equal weight to equal reward trajectories

  trans_mat: an S x A x S' array of transition probabilites from state s to s' if action a is taken
  horizon: the finite time horizon (int) of the problem for calculating state frequencies
  r_weights: a size F array of the weights of the current reward function to evaluate
  state_features: an S x F array that lists F feature values for each state in S

  return: an S x A policy in which each entry is the probability of taking action a in state s
  """
  n_states = np.shape(trans_mat)[0]
  n_actions = np.shape(trans_mat)[1]
  policy = np.zeros((n_states,n_actions))
  value_function = np.zeros(n_states)
  q_function = np.zeros((n_states,n_actions))
  reward = np.matmul(state_features,r_weights)
#Value Iteration
  for i in range(100):
    for s in range(25):
      for s_prime in range(25):
        for a in range(n_actions):
          #import pdb;pdb.set_trace()
          if s!= 26 and s_prime!=26:
            q_function[s,a] = trans_mat[s,a,s_prime]*(reward[s]+((0.99)*value_function[s+1]))
      value_function[s] = max(q_function[s])
#Policy Updating
  for s in range(n_states):
    for a in range(n_actions):
      policy[s,a] = np.argmax(q_function[s,a])
  return policy

def calcExpectedStateFreq(trans_mat, horizon, start_dist, policy):
  """
  Given a MaxEnt policy, begin with the start state distribution and propagate forward to find the expected state frequencies over the horizon

  trans_mat: an S x A x S' array of transition probabilites from state s to s' if action a is taken
  horizon: the finite time horizon (int) of the problem for calculating state frequencies
  start_dist: a size S array of starting start probabilities - must sum to 1
  policy: an S x A array array of probabilities of taking action a when in state s

  return: a size S array of expected state visitation frequencies

  Algorithm

  SVF
  start distribution  = 1 for first state and 0 for everything else
  for s in S
    dt(0,s) = start_distribution[s]
    for t in T
        for a in A
            for all s_prime in S
                dt+1[s] = dt[t,s_prime]*policy[s_prime/a]*trans[s_prime,s,a]
        dt[s,t+1] = dt
  """
  n_states = np.shape(trans_mat)[0]
  n_actions = np.shape(trans_mat)[1]
  state_freq = np.zeros(len(start_dist))
  dt = np.zeros((horizon,n_states))
  for s in range(n_states):
    dt[0,s] = start_dist[s]
    for t in range(horizon-1):
      for a in range(n_actions):
        for s_prime in range(n_states):
          dt[t+1,s] += dt[t,s_prime]*policy[s_prime,a]*trans_mat[s_prime,a,s]
  state_freq = np.sum(dt,0)
  return state_freq

def maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon, learning_rate):
  """
  Compute a MaxEnt reward function from demonstration trajectories

  trans_mat: an S x A x S' array that describes transition probabilites from state s to s' if action a is taken
  state_features: an S x F array that lists F feature values for each state in S
  demos: a list of lists containing D demos of varying lengths, where each demo is series of states (ints)
  seed_weights: a size F array of starting reward weights
  n_epochs: how many times (int) to perform gradient descent steps
  horizon: the finite time horizon (int) of the problem for calculating state frequencies
  learning_rate: a multiplicative factor (float) that determines gradient step size

  return: a size F array of reward weights
  """
  n_states = np.shape(trans_mat)[0]
  n_features = np.shape(state_features)[1]
  r_weights = np.zeros(n_features)
  start_dist = np.zeros(np.shape(trans_mat)[0])
  start_dist[0] = 1
  f = 0
  for d in demos:
    for s in d:
      f += state_features[s]
  f_hat = f/len(demos)

  for i in range(n_epochs):
    policy = calcMaxEntPolicy(trans_mat, horizon, r_weights, state_features)
    svf = calcExpectedStateFreq(trans_mat, horizon, start_dist, policy)
    gradient = f_hat - np.matmul(svf,state_features)
    r_weights = r_weights+(0.0001*gradient)
  return r_weights

def main():

  # Build domain, features, and demos
  trans_mat = build_trans_mat_gridworld()
  state_features = np.eye(26,25)  # Terminal state has no features, forcing zero reward
  demos = [[0,1,2,3,4,9,14,19,24,25],[0,5,10,15,20,21,22,23,24,25],[0,5,6,11,12,17,18,23,24,25],[0,1,6,7,12,13,18,19,24,25]]
  seed_weights = np.zeros(25)

  # Parameters
  n_epochs = 100
  horizon = 10
  learning_rate = 1.0
  r_weights = maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon, learning_rate)
    # Construct reward function from weights and state features
  reward_fxn = []
  for s_i in range(25):
    reward_fxn.append( np.dot(r_weights, state_features[s_i]) )
  reward_fxn = np.reshape(reward_fxn, (5,5))
  print(r_weights)
  # Plot reward function
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  X = np.arange(0, 5, 1)
  Y = np.arange(0, 5, 1)
  X, Y = np.meshgrid(X, Y)
  surf = ax.plot_surface(X, Y, reward_fxn, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
  plt.show()
main()