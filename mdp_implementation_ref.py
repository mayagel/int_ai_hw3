from copy import deepcopy
import numpy as np
from mdp import MDP
from termcolor import colored


def get_legal_states(mdp):
    legal_states = []
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] == "WALL" or (i, j) in mdp.terminal_states:
                continue
            legal_states.append((i, j))
    return legal_states


def get_states(mdp: MDP):
    states = []
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] == "WALL":
                continue
            states.append((i, j))
    return states


def get_utility(U, state):
    i, j = state
    return U[i][j]


def get_rewards(mdp: MDP, state):
    i, j = state
    return float(mdp.board[i][j])


def utility(mdp: MDP, U, state, action):
    actions = ["UP", "DOWN", "RIGHT", "LEFT"]
    value = []
    util = 0
    prob = mdp.transition_function[action]
    for a in actions:
        i, j = mdp.step(state, a)
        value.append(U[i][j])
    for p, val in zip(prob, value):
        util += p * val
    return util


def get_action_from_policy(policy, state):
    i, j = state
    return policy[i][j]


def transition(mdp: MDP, state, destination, action):
    actions = ["UP", "DOWN", "RIGHT", "LEFT"]
    valid_actions = []
    p = 0
    if state in mdp.terminal_states:
        return 0
    for a in actions:
        if mdp.step(state, a) == destination:
            valid_actions.append(a)
    for a in valid_actions:
        p += mdp.transition_function[action][actions.index(a)]
    return p


def get_total_utility(mdp: MDP, U, state, action):
    utils = [transition(mdp, state, destination, action) * get_utility(U, destination)
             for destination in get_states(mdp)]
    return sum(utils)


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    actions = ["UP", "DOWN", "RIGHT", "LEFT"]
    U = None
    U_tag = deepcopy(U_init)
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] == "WALL":
                U_tag[i][j] = None
    for i, j in mdp.terminal_states:
        U_tag[i][j] = float(mdp.board[i][j])
    delta = np.inf
    gamma = mdp.gamma
    while delta > (epsilon * (1 - gamma))/gamma:
        delta = 0
        U = deepcopy(U_tag)
        legal_states = get_legal_states(mdp)
        for i, j in legal_states:
            values = []
            reward = float(mdp.board[i][j])
            for a in actions:
                values.append(utility(mdp, U, (i, j), a))
            U_tag[i][j] = reward + gamma * max(values)
            delta = max(delta, abs(U[i][j] - U_tag[i][j]))
    return U
    # ========================


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    actions = ["UP", "DOWN", "RIGHT", "LEFT"]
    values = []
    policy = deepcopy(U)
    legal_states = get_legal_states(mdp)
    for i, j in legal_states:
        values = []
        for a in actions:
            values.append(utility(mdp, U, (i, j), a))
        index = values.index(max(values))
        policy[i][j] = actions[index]

    return policy
    # ========================


def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    rewards_vec = np.array([get_rewards(mdp, state) for state in get_states(mdp)])
    transitions_vec = np.array([[transition(mdp, state, destination, get_action_from_policy(policy, state))
                                for destination in get_states(mdp)] for state in get_states(mdp)])

    utility = np.linalg.inv(np.eye(len(rewards_vec)) - mdp.gamma * transitions_vec) @ rewards_vec

    U= deepcopy(policy)
    for state, u in zip(get_states(mdp), utility.tolist()):
        i, j = state
        U[i][j] = u
    return U
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    actions = ["UP", "DOWN", "RIGHT", "LEFT"]
    policy = deepcopy(policy_init)
    changed = True
    while changed:
        U = policy_evaluation(mdp, policy)
        changed = False
        for state in get_states(mdp):
            temp_util = [get_total_utility(mdp, U, state, a) for a in actions]
            max_util = max(temp_util)
            action = get_action_from_policy(policy, state)
            policy_util = get_total_utility(mdp, U, state, action)
            if max_util > policy_util:
                i, j = state
                policy[i][j] = actions[temp_util.index(max_util)]
                changed = True
    return policy
    # ========================



"""For this functions, you can import what ever you want """
import math


def get_utility_for_state(mdp, state, U):
    actions = ["UP", "DOWN", "RIGHT", "LEFT"]
    i, j = state
    current_max = -np.inf
    for desire_action in actions:
        sum_value = 0
        for index, actual_action in enumerate(mdp.actions.keys()):
            i_, j_ = mdp.step((i, j), actual_action)
            utils = float(U[i_][j_] * mdp.transition_function[desire_action][index])
            sum_value += utils
        current_max = max(current_max, sum_value)
    return current_max


def get_utility_for_desire_action(mdp, state, U, action):
    sum_value = 0
    i, j = state
    for index, actual_action in enumerate(mdp.actions.keys()):
        i_, j_ = mdp.step((i, j), actual_action)
        utils = float(U[i_][j_] * mdp.transition_function[action][index])
        sum_value += utils
    return sum_value


def get_policies(mdp, U, epsilon):
    actions = ["UP", "DOWN", "RIGHT", "LEFT"]
    digits = len(str(epsilon).split(".")[1]) + 1
    U_ = np.full((mdp.num_row, mdp.num_col), None, dtype=object)
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] == "WALL":
                continue
            if (i, j) not in mdp.terminal_states:
                utility_for_state = get_utility_for_state(mdp, (i, j), U)
                policy_actions = []
                for desire_action in actions:
                    utility_for_desire_action = get_utility_for_desire_action(mdp, (i, j), U, desire_action)
                    if abs(round(utility_for_desire_action, digits) - round(utility_for_state, digits)) < epsilon:
                        policy_actions.append(desire_action)
                U_[i][j] = policy_actions
    return U_


def get_longest_string(mdp, policies):
    longest_str = 0
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if policies[i][j] is not None:
                longest_str = max(longest_str, len(policies[i][j]))
    return longest_str


def print_multi_policy(mdp, policy):
    direction = {"UP": u'\u2191', 'DOWN': u'\u2193', 'RIGHT': u'\u2192', 'LEFT': u'\u2190'}
    res = ""
    for r in range(mdp.num_row):
        res += "|"
        for c in range(mdp.num_col):
            if mdp.board[r][c] == 'WALL' or (r, c) in mdp.terminal_states:
                val = mdp.board[r][c]
            else:
                val = ""
                for item in policy[r][c]:
                    val += direction[item]
            if (r, c) in mdp.terminal_states:
                res += " " + colored(val[:5].ljust(5), 'red') + " |"  # format
            elif mdp.board[r][c] == 'WALL':
                res += " " + colored(val[:5].ljust(5), 'blue') + " |"  # format
            else:
                res += " " + val[:5].ljust(5) + " |"  # format
        res += "\n"
    print(res)


def get_number_of_policies(mdp, policies):
    sum_policies = 1
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] != "WALL" and (i, j) not in mdp.terminal_states:
                curren_policy = policies[i][j]
                sum_policies *= len(curren_policy)
    return sum_policies


def get_all_policies(mdp, U, epsilon=10**(-3), return_policies=False, print_number_of_policies=False):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp, and the utility value U (which satisfies the Belman equation)
    # print / display all the policies that maintain this value
    # (a visualization must be performed to display all the policies)
    #
    # return: the number of different policies
    #

    # ====== YOUR CODE: ======
    policies = get_policies(mdp, U, epsilon)
    if return_policies:
        return policies
    else:
        print_multi_policy(mdp, policies)
    number_of_policies = get_number_of_policies(mdp, policies)
    if print_number_of_policies:
        print(f'\n Number of policies: {number_of_policies}')
    return number_of_policies
    # ========================


def does_policies_equal(policy1, policy2):
    r_policy_1 = len(policy1)
    c_policy_1 = len(policy1[0])
    r_policy_2 = len(policy2)
    c_policy_2 = len(policy2[0])
    if r_policy_1 != r_policy_2 or c_policy_1 != c_policy_2:
        return False
    for i in range(len(policy1)):
        for j in range(len(policy1[i])):
            if policy1[i][j] != policy2[i][j]:
                return False
    return True


def get_policy_for_different_rewards(mdp, epsilon=10**(-3)):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp
    # print / displas the optimal policy as a function of r
    # (reward values for any non-finite state)
    #
    # ====== YOUR CODE: ======
    policies_list = []
    board_init = deepcopy(mdp.board)
    reward = np.arange(-5, 5.01, 0.01).tolist()
    for index, R in enumerate(reward):
        reward[index] = math.ceil(R * 100)/100
    for R in reward:
        for i in range(mdp.num_row):
            for j in range(mdp.num_col):
                if mdp.board[i][j] != 'WALL' and (i, j) not in mdp.terminal_states:
                    mdp.board[i][j] = R
        U_tag = []
        for i in range(mdp.num_row):
            U_tag.append([])
            for j in range(mdp.num_col):
                U_tag[-1].append(0)
        U = value_iteration(mdp, U_tag)
        policies = get_all_policies(mdp, U, epsilon, True)
        if len(policies_list) == 0:
            policies_list.append([policies, R, None])
        elif not does_policies_equal(policies_list[-1][0], policies):
            policies_list[-1] = [policies_list[-1][0], policies_list[-1][1], R]
            policies_list.append([policies, policies_list[-1][2], None])
        mdp.board = board_init
    policies_list[-1] = [policies_list[-1][0], policies_list[-1][1], 5]
    reward_changed_policy = []
    for policy in policies_list:
        print_multi_policy(mdp, policy[0])
        if policy[1] == -5.0:
            print(f'R(s) < {policy[2]} \n \n')
        elif policy[2] == 5.0:
            print(f'R(s) >= {policy[1]} \n \n')
        else:
            print(f'{policy[1]} <= R(s) < {policy[2]} \n \n')
        reward_changed_policy.append(policy[2])
    reward_changed_policy.pop(-1)
    print(f'Rewards at which policy changed:\n{reward_changed_policy}')
    return reward_changed_policy
    #========================


