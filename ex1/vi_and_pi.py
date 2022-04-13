# MDP Value Iteration and Policy Iteration
import argparse
import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

parser = argparse.ArgumentParser(description='A program to run assignment 1 implementations.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--env",
                    help="The name of the environment to run your algorithm on.",
                    choices=["Deterministic-4x4-FrozenLake-v0", "Stochastic-4x4-FrozenLake-v0"],
                    default="Deterministic-4x4-FrozenLake-v0")

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

P: nested dictionary
    From gym.core.Environment
    For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
    tuple of the form (probability, nextstate, reward, terminal) where
        - probability: float
            the probability of transitioning from "state" to "nextstate" with "action"
        - nextstate: int
            denotes the state we transition to (in range [0, nS - 1])
        - reward: int
            either 0 or 1, the reward for transitioning from "state" to
            "nextstate" with "action"
        - terminal: bool
          True when "nextstate" is a terminal state (hole or goal), False otherwise
nS: int
    number of states in the environment
nA: int
    number of actions in the environment
gamma: float
    Discount factor. Number in range [0, 1)
"""
PROB, NEXT_STG, REWARD, TERMINAL = 0, 1, 2, 3  # meaningful indexing of P's values


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """Evaluate the value function from a given policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns
    -------
    # we implement the bellman backup for a particular policy
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """
    prev_value_function = np.ones(nS) * -np.inf
    value_function = np.zeros(nS)
    while True:
        for state, action in enumerate(policy):
            greedy_term = 0
            immediate_reward = P[state][action][0][REWARD]  # before loop as all the rewards in P[s][a] are the same
            for prob, next_state, reward, terminal in P[state][action]:  # all possible s' from s with action a
                greedy_term += prob * value_function[next_state]
            value_function[state] = immediate_reward + gamma * greedy_term
        if np.linalg.norm(value_function - prev_value_function, np.inf) <= tol:
            break
        else:
            prev_value_function = value_function
    return value_function


def policy_improvement(P, nS: int, nA: int, value_from_policy: np.ndarray, policy: np.array, gamma: float = 0.9):
    """Given the value function from policy improve the policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    policy: np.array
        The previous policy. ( # I  dont think we need this, as we have V_pi )

    Returns
    -------
    new_policy: np.ndarray[nS]
        An array of integers. Each integer is the optimal action to take
        in that state according to the environment dynamics and the
        given value function.
    """

    new_policy = np.zeros(nS, dtype='int')
    ############################
    # YOUR IMPLEMENTATION HERE #
    for state in range(nS):
        state_action_value = np.zeros(nA)  # Q_pi(s,a)
        for action in range(nA):
            greedy_term = 0
            immediate_reward = P[state][action][0][REWARD]  # before loop as all the rewards in P[s][a] are the same
            for prob, next_state, reward, terminal in P[state][action]:  # all possible s' from s with action a
                greedy_term += prob * value_from_policy[next_state]
            state_action_value[action] = immediate_reward + gamma * greedy_term
        new_policy[state] = np.argmax(state_action_value)
    ############################
    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    while True:
        curr_value_function = policy_evaluation(P, nS, nA, policy)
        curr_policy = policy_improvement(P, nS, nA, curr_value_function, policy)
        if (curr_policy == policy).all():
            break
        else:
            policy = curr_policy
    value_function = curr_value_function
    ############################
    return value_function, policy


def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    v_tag = np.zeros(nS)
    v = np.ones(nS) * np.inf
    # policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    while np.linalg.norm(v - v_tag, np.inf) > tol:
        v = v_tag
        # list comprehension is fun!
        v_tag = [np.max(
            [P[s][a][0][REWARD] + gamma * np.sum([pr * v_tag[s_tag] for pr, s_tag, _, _ in P[s][a]])
             for a in range(nA)]) for s in range(nS)]
    v_star = v
    pi_star = [np.argmax(
        [P[s][a][0][REWARD] + gamma * np.sum([pr * v_star[s_tag] for pr, s_tag, _, _ in P[s][a]])
         for a in range(nA)]) for s in range(nS)]
    ############################
    return v_tag, pi_star


def render_single(env, policy, max_steps=100):
    """
      This function does not need to be modified
      Renders policy once on environment. Watch your agent play!

      Parameters
      ----------
      env: gym.core.Environment
        Environment to play on. Must have nS, nA, and P as
        attributes.
      Policy: np.array of shape [env.nS]
        The action to take at a given state
    """

    episode_reward = 0
    ob = env.reset()
    for t in range(max_steps):
        env.render()
        time.sleep(0.25)
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    env.render();
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":
    # read in script argument
    args = parser.parse_args()

    # Make gym environment
    env = gym.make(args.env)

    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_pi, 100)

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_vi, 100)
