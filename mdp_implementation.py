from copy import deepcopy
import numpy as np
import mdp

actions_dict = {
    'UP': 0,
    'DOWN': 1,
    'RIGHT': 2,
    'LEFT': 3
    }

def calc_action(mdp, U, x, y, action):
    # res = 0
    # for a in mdp.actions:
    #     (new_x, new_y) = mdp.step((x, y), a)
    #     res += (mdp.transition_function[action])[actions_dict[a]] * U[new_x][new_y]
    # return res
    return sum([(mdp.transition_function[action])[actions_dict[a]] * U[mdp.step((x, y), a)[0]][mdp.step((x, y), a)[1]] for a in mdp.actions])

def max_u(mdp, U, x, y):
    max_val = max([round(calc_action(mdp, U, x, y, action), 2) for action in mdp.actions])
    return float(mdp.board[x][y] + (max_val * mdp.gamma))


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    U_res = None
    U_org = deepcopy(U_init)
    for x, y in mdp.terminal_states:
        U_org[x][y] = float(mdp.board[x][y])
    delta = float('inf')
    coordinates = [(x, y) for x in range(mdp.num_row) for y in range(mdp.num_col)]
    while delta > epsilon * (1 - mdp.gamma) / mdp.gamma:
        U_res = deepcopy(U_org)
        delta = 0
        for x, y in coordinates:
            if mdp.board[x][y] in mdp.terminal_states or mdp.board[x][y] == 'WALL':
                continue
            U_org[x][y] = max_u(mdp, U_org, x, y)
            delta = max(delta, abs(U_res[x][y] - U_org[x][y]))
    U_res = deepcopy(U_org)
    return U_res


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================



""" For these functions, you can import what ever you want """


def get_all_policies(mdp, U, epsilon=10 ** (-3)):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp, and the utility value U (which satisfies the Belman equation)
    # print / display all the policies that maintain this value
    # (a visualization must be performed to display all the policies)
    #
    # return: the number of different policies
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def get_policy_for_different_rewards(mdp, epsilon=10 ** (-3)):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp
    # print / displas the optimal policy as a function of r
    # (reward values for any non-finite state)
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
