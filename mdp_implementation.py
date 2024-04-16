from copy import deepcopy
import numpy as np
import mdp
from termcolor import colored

actions_dict = {
    'UP': 0,
    'DOWN': 1,
    'RIGHT': 2,
    'LEFT': 3
    }

directions_dict = {'UP': u'\u2191',
                  'DOWN': u'\u2193',
                  'RIGHT': u'\u2192',
                  'LEFT': u'\u2190'
                  }


def get_coordinates(mdp):
    return [(x, y) for x in range(mdp.num_row) for y in range(mdp.num_col)
            if (x,y) not in mdp.terminal_states and mdp.board[x][y] != 'WALL']

def calc_action(mdp, U, x, y, action):
    return sum([mdp.transition_function[action][actions_dict[a]] * U[mdp.step((x, y), a)[0]][mdp.step((x, y), a)[1]] for a in mdp.actions])

def max_action(mdp, U, x, y):
    actions_val = {a: calc_action(mdp, U, x, y, a) for a in mdp.actions}
    # actions_val = {}
    # for a in mdp.actions:
    #     actions_val[a] = calc_action(mdp, U, x, y, a)
    max_a = max(actions_val, key=actions_val.get)
    return (max_a, actions_val[max_a])

def max_u(mdp, U, x, y):
    # max_val = max([round(calc_action(mdp, U, x, y, a), 2) for a in mdp.actions])
    # max_val = round(max(mdp.actions, key=lambda a: calc_action(mdp, U, x, y, a)), 2)
    max_val = round(max_action(mdp, U, x, y)[1], 2)
    return float(float(mdp.board[x][y]) + (max_val * mdp.gamma))


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    U_res = None
    U_org = deepcopy(U_init)
    for x, y in mdp.terminal_states:
        U_org[x][y] = float(mdp.board[x][y])
    delta = float('inf')
    coordinates = get_coordinates(mdp)
    while delta > epsilon * (1 - mdp.gamma) / mdp.gamma:
        U_res = deepcopy(U_org)
        delta = 0
        for x, y in coordinates:
            U_org[x][y] = max_u(mdp, U_org, x, y)
            delta = max(delta, abs(U_res[x][y] - U_org[x][y]))
    U_res = deepcopy(U_org)
    return U_res


def get_policy(mdp, U):
    policy = [[""] * mdp.num_col for _ in range(mdp.num_row)]
    coordinates = get_coordinates(mdp)
    for x, y in coordinates:
        policy[x][y] = max(mdp.actions, key=lambda a: calc_action(mdp, U, x, y, a))
    return policy


def policy_evaluation(mdp, policy):
    coordinates = get_coordinates(mdp)
    I = np.eye(mdp.num_row * mdp.num_col)
    policy_mat = np.zeros((mdp.num_row * mdp.num_col, mdp.num_row * mdp.num_col))
    reward = np.zeros(mdp.num_row * mdp.num_col)
    for x,y in coordinates:
        reward[x * mdp.num_col + y] = float(mdp.board[x][y])
        action = policy[x][y]
        for a in mdp.actions:
            n_state = mdp.step((x, y), a)
            n_state_idx = n_state[0] * mdp.num_col + n_state[1]
            policy_mat[x * mdp.num_col + y][n_state_idx] += mdp.transition_function[action][actions_dict[a]]
    mat_sum = np.add(np.dot(-mdp.gamma, policy_mat), I)
    return np.linalg.solve(mat_sum, reward).reshape((mdp.num_row, mdp.num_col))


def policy_iteration(mdp, policy_init):
    coordinates = get_coordinates(mdp)
    modified = True
    while modified:
        U = policy_evaluation(mdp, policy_init)
        modified = False
        for x, y in coordinates:
            max_value = calc_action(mdp, U, x, y, max_action(mdp, U, x, y)[0])
            predicted_value = calc_action(mdp, U, x, y, policy_init[x][y])
            if round(max_value, 2) > round(predicted_value, 2):
                policy_init[x][y] = max_action(mdp, U, x, y)[0]
                modified = True
    return policy_init


""" For these functions, you can import what ever you want """
import math

def calc_policies(mdp, U, epsilon):
    accuracy = len(str(epsilon)[str(epsilon).find(".") + 1:]) + 1
    policies = np.full((mdp.num_row, mdp.num_col), None, dtype=object)
    policies_counter = 1
    for x, y in get_coordinates(mdp):
        policies[x][y] = [a for a in mdp.actions.keys() 
                          if round(calc_action(mdp, U, x, y, a), accuracy) - round(max_action(mdp, U, x, y)[1], accuracy) < epsilon]
        policies_counter *= len(policies[x][y])
    return (policies, policies_counter)

def print_policies(mdp, policy):
    res = ""
    for x in range(mdp.num_row):
        res += "|"
        for y in range(mdp.num_col):
            if (x, y) in mdp.terminal_states:
                res += " " + colored(mdp.board[x][y][:5].ljust(5), 'red') + " |"
            elif mdp.board[x][y] == 'WALL':
                res += " " + colored(mdp.board[x][y][:5].ljust(5), 'blue') + " |"
            else:
                val = ""
                for a in policy[x][y]:
                    val += directions_dict[a]
                res += " " + val[:5].ljust(5) + " |"
        res += "\n"
    print(res)

def get_all_policies(mdp, U, epsilon=10 ** (-3), ret_policies=False, print_policies_num=False):  # You can add more input parameters as needed
    policies, policies_num = calc_policies(mdp, U, epsilon)
    if ret_policies:
        return policies
    print_policies(mdp, policies)
    if print_policies_num:
        print(f'\n Number of policies: {policies_num}')
    return policies_num

def equal_policies(p1, p2):
    if len(p1) != len(p2) or len(p1[0]) != len(p2[0]):
        return False
    for x, y in [(i, j) for i in range(len(p1)) for j in range(len(p1[0]))]:
        if p1[x][y] != p2[x][y]:
            return False
    return True

def get_policy_for_different_rewards(mdp, epsilon=10 ** (-3)):  # You can add more input parameters as needed
    policies_arr = []
    board = deepcopy(mdp.board)
    rewards = [math.ceil(r * 100) / 100 for r in np.arange(-5, 5.01, 0.01)]

    for r in rewards:
        for x, y in get_coordinates(mdp):
            mdp.board[x][y] = r
        U = value_iteration(mdp, [[0 for _ in range(mdp.num_col)] for _ in range(mdp.num_row)])
        policies = get_all_policies(mdp, U, epsilon, True)
        if not policies_arr:
            policies_arr.append([policies, r, None])
        elif not equal_policies(policies_arr[-1][0], policies):
            policies_arr[-1][2] = r
            policies_arr.append([policies, r, None])
        mdp.board = board
    policies_arr[-1][2] = 5

    for policy in policies_arr:
        print_policies(mdp, policy[0])
        if policy[1] == -5.0:
            print(f'R(s) < {policy[2]} \n \n')
        elif policy[2] == 5.0:
            print(f'R(s) >= {policy[1]} \n \n')
        else:
            print(f'{policy[1]} <= R(s) < {policy[2]} \n \n')

    rewards_res = [p[2] for p in policies_arr].pop(-1)
    print(f'Rewards at which policy changed:\n{rewards_res}')
    return rewards_res
