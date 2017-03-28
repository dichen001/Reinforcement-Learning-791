import collections
import numpy as np
import pandas as pd
from itertools import permutations
import mdptoolbox, mdptoolbox.example
import argparse


def generate_MDP_input2(original_data, features):

    students_variables = ['student', 'priorTutorAction', 'reward']

    # generate distinct state based on feature
    # shitian's code is too slow for converting into strings.
    # original_data['state'] = original_data[features].apply(lambda x: ':'.join(str(v) for v in x), axis=1)
    original_data['state'] = original_data[features].apply(tuple, axis=1)
    students_variables = students_variables + ['state']
    data = original_data[students_variables]

    # quantify actions
    distinct_acts = list(data['priorTutorAction'].unique())
    Nx = len(distinct_acts)
    i = 0
    for act in distinct_acts:
        data.loc[data['priorTutorAction'] == act, 'priorTutorAction'] = i
        i += 1

    # initialize state transition table, expected reward table, starting state table
    distinct_states = list(data['state'].unique())
    Ns = len(distinct_states)
    start_states = np.zeros(Ns)
    A = np.zeros((Nx, Ns, Ns))
    expectR = np.zeros((Nx, Ns, Ns))

    # update table values episode by episode
    # each episode is a student data set
    student_list = list(data['student'].unique())
    for student in student_list:
        student_data = data.loc[data['student'] == student,]
        row_list = student_data.index.tolist()

        # count the number of start state
        start_states[distinct_states.index(student_data.loc[row_list[0], 'state'])] += 1

        for i in range(1, len(row_list)):
            state1 = distinct_states.index(student_data.loc[row_list[i - 1], 'state'])
            state2 = distinct_states.index(student_data.loc[row_list[i], 'state'])
            act = student_data.loc[row_list[i], 'priorTutorAction']

            # count the state transition
            A[act, state1, state2] += 1
            expectR[act, state1, state2] += float(student_data.loc[row_list[i], 'reward'])

    # normalization
    start_states = start_states / np.sum(start_states)

    for act in range(Nx):
        # generate expected reward
        # it has the warning, ignore it
        with np.errstate(divide='ignore', invalid='ignore'):
            expectR[act] = np.divide(expectR[act], A[act])
            expectR[np.isnan(expectR)] = 0

        # each column will sum to 1 for each row, obtain the state transition table
        A[act] = np.divide(A[act].transpose(), np.sum(A[act], axis=1))
        A[act] = A[act].transpose()

    return [start_states, A, expectR, distinct_acts, distinct_states]


def calcuate_ECR(start_states, expectV):
    ECR_value = start_states.dot(np.array(expectV))
    return ECR_value


def output_policy(distinct_acts, distinct_states, vi):
    Ns = len(distinct_states)
    print('Policy: ')
    print('state -> action, value-function')
    for s in range(Ns):
        # print(distinct_states[s] + " -> " + distinct_acts[vi.policy[s]] + ", " + str(vi.V[s]))
        print(str(distinct_states[s]) + " -> " + distinct_acts[vi.policy[s]] + ", " + str(vi.V[s]))


def induce_policy_MDP2(original_data, selected_features):

    [start_states, A, expectR, distinct_acts, distinct_states] = generate_MDP_input2(original_data, selected_features)

    # apply Value Iteration to run the MDP
    vi = mdptoolbox.mdp.ValueIteration(A, expectR, 0.9)
    vi.run()

    # output policy
    output_policy(distinct_acts, distinct_states, vi)

    # evaluate policy using ECR
    ECR_value = calcuate_ECR(start_states, vi.V)
    print('ECR value: ' + str(ECR_value))
    return ECR_value


def pct_rank_qcut(series, n):
    edges = pd.Series([float(i) / n for i in range(n + 1)])
    f = lambda x: (edges >= x).argmax()
    return series.rank(pct=1).apply(f)


if __name__ == "__main__":

    # original_data = pandas.read_csv('MDP_training_data.csv')
    od = pd.read_csv('MDP_Original_data.csv')
    headers = list(od.columns.values)
    staticHeader, allFeatures = headers[:6], headers[6:]
    for f in allFeatures:
        if type(od.loc[0, f]) == type(np.float64(1.11)):
            # discretize
            bins = min(100, len(od[f].unique()))
            # print f + ':\t' + str(bins)
            # od[f] = pct_rank_qcut(od[f], bins)
            od[f] = pd.cut(od[f], bins, right=True, labels=range(bins))

    IterNum, limit = 10, 8
    record = {}
    for i in range(IterNum):
        unseen = set(allFeatures)
        selected_features = []
        prior_ECR = 0
        print 'iteration ' + str(i)
        while unseen:
            # random add one
            selected_features += [unseen.pop()]
            cur_ECR = induce_policy_MDP2(od, selected_features)
            # no improvement, jump
            if cur_ECR < prior_ECR:
                print 'jump ' + selected_features[-1]
                selected_features.pop()
                continue
            # improved --> add!
            elif len(selected_features) > limit:
                print 'overlimit'
                # over the limit of features, find one to abandon!
                for j, sf in enumerate(selected_features):
                    tmp_removed_features = selected_features[0:j] + selected_features[j+1:]
                    tmp_ECR = induce_policy_MDP2(od, tmp_removed_features)
                    # found the one to abandon
                    if tmp_ECR > cur_ECR:
                        print 'remove '  + sf
                        selected_features = tmp_removed_features
                        cur_ECR = tmp_ECR
                        break
                # can't find one to abandon to improve the ECR
                if j >= limit:
                    print 'can\'t find one to abandon'
                    break
                    # another strategy is to abondon the lowest ECR
            print 'add ' + selected_features[-1]
            prior_ECR = cur_ECR
        record[', '.join(selected_features)] = prior_ECR







    selected_features = ['Level', 'probDiff']
    ECR_value = induce_policy_MDP2(od, selected_features)
