from copy import deepcopy
import numpy as np
import mdp

actions_dict = {
    'UP': 0,
    'DOWN': 1,
    'RIGHT': 2,
    'LEFT': 3
    }

def get_coordinates(mdp):
    return [(x, y) for x in range(mdp.num_row) for y in range(mdp.num_col)
            if mdp.board[x][y] not in mdp.terminal_states and mdp.board[x][y] != 'WALL']

def calc_action(mdp, U, x, y, action):
    # return sum([mdp.transition_function[action][actions_dict[a]] *
    #             U[mdp.step((x, y), a)[0]][mdp.step((x, y), a)[1]] for a in mdp.actions])
    return sum(mdp.actions, key=lambda a: mdp.transition_function[action][actions_dict[a]] *
               U[mdp.step((x, y), a)[0]][mdp.step((x, y), a)[1]])

def max_action(mdp, U, x, y):
    return max(mdp.actions, key=lambda a: calc_action(mdp, U, x, y, a))

def max_u(mdp, U, x, y):
    # max_val = max([round(calc_action(mdp, U, x, y, a), 2) for a in mdp.actions])
    # max_val = round(max(mdp.actions, key=lambda a: calc_action(mdp, U, x, y, a)), 2)
    max_val = round(max_action(mdp, U, x, y), 2)
    return float(mdp.board[x][y] + (max_val * mdp.gamma))


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
    # policy = [[""] * mdp.num_col for _ in range(mdp.num_row)]
    # coordinates = [(x, y) for x in range(mdp.num_row) for y in range(mdp.num_col)]
    # for x, y in coordinates:
    #     if mdp.board[x][y] in mdp.terminal_states or mdp.board[x][y] == 'WALL':
    #         continue
    #     policy[x][y] = max(mdp.actions, key=lambda a: calc_action(mdp, U, x, y, a))
    # # policy = [max(mdp.actions, key=lambda a: calc_action(mdp, U, x, y, a))
    # #           for x in range(mdp.num_row)
    # #           for y in range(mdp.num_col)
    # #           if mdp.board[x][y] not in mdp.terminal_states and mdp.board[x][y] != 'WALL']
    policy = [max_action(mdp, U, x, y)
              for x in range(mdp.num_row)
              for y in range(mdp.num_col)
              if mdp.board[x][y] not in mdp.terminal_states and mdp.board[x][y] != 'WALL']
    return policy


def policy_evaluation(mdp, policy):
    coordinates = get_coordinates(mdp)
    I = np.eye(mdp.num_row * mdp.num_col)
    policy_mat = np.zeros((mdp.num_row * mdp.num_col, mdp.num_row * mdp.num_col))
    reward = np.zeros((mdp.num_row, mdp.num_col))
    for x,y in coordinates:
        reward[x][y] = float(mdp.board[x][y])
        action = policy[x][y]
        for a in mdp.actions:
            n_state = mdp.step((x, y), a)
            n_state_idx = n_state[0] * mdp.num_col + n_state[1]
            policy_mat[x * mdp.num_col + y][n_state_idx] += mdp.transition_function[action][actions_dict[a]]
    mat_sum = np.add(I, np.dot(-mdp.gamma, policy_mat))
    # mat_sum = I - mdp.gamma @ policy_mat
    return np.linalg.solve(mat_sum, reward).reshape((mdp.num_row, mdp.num_col))


def policy_iteration(mdp, policy_init):
    coordinates = get_coordinates(mdp)
    modified = True
    while modified:
        U = policy_evaluation(mdp, policy_init)
        modified = False
        for x, y in coordinates:
            max_value = calc_action(mdp, U, x, y, max_action(mdp, U, x, y))
            predicted_value = calc_action(mdp, U, x, y, policy_init[x][y])
            if round(max_value, 2) > round(predicted_value, 2):
                policy_init[x][y] = max_action(mdp, U, x, y)
                modified = True
    return policy_init


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
