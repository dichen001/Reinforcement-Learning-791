import collections
import numpy as np
import pandas as pd
import pandas.core.algorithms as algos
from itertools import permutations
import mdptoolbox, mdptoolbox.example
import argparse
from random import shuffle


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
        check = np.where(np.sum(A[act],axis=1)==0)[0]
        # print check
        for l in check:
            A[act][l][l] = 1
        # check = np.where(np.sum(A[act],axis=1)==0)[0]
        # print check
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
    vi =  mdptoolbox.mdp.ValueIteration(A, expectR, 0.9)
    vi.run()

    # output policy
    # output_policy(distinct_acts, distinct_states, vi)

    # evaluate policy using ECR
    ECR_value = calcuate_ECR(start_states, vi.V)
    print('ECR value: ' + str(ECR_value))
    return ECR_value


def pct_rank_qcut(series, n):
    edges = pd.Series([float(i) / n for i in range(n + 1)])
    f = lambda x: (edges >= x).argmax()
    return series.rank(pct=1).apply(f)


# def discretization(features, data, bins=10):
#     for f in allFeatures:
#         if len(data[f].unique()) > bins:
#             data[f] = pd.cut(data[f], bins, right=True, labels=range(bins))
#         # if type(data.loc[0, f]) == type(np.float64(1.11)):
#         #     # discretize
#         #     bins = min(10, len(data[f].unique()))
#         #     # print f + ':\t' + str(bins)
#         #     # od[f] = pct_rank_qcut(od[f], bins)
#         #     data[f] = pandas.cut(data[f], bins, right=True, labels=range(bins))



def discretization(features, data, bins=10):
    for f in allFeatures:
        uniques = data[f].unique()
        l = len(uniques)
        if l > bins:
            # data[f] = pandas.cut(data[f], bins, labels=range(bins))
            # data[f] = pandas.qcut(data[f] + jitter(data[f]), 2, labels=range(2))
            ranges = algos.quantile(uniques, np.linspace(0, 1, 3))
            result = pd.tools.tile._bins_to_cuts(data[f], ranges, include_lowest=True)
            print result.value_counts()
            range_names = list(result.values.unique())
            data[f]= result.apply(lambda x : range_names.index(x))
            # print '\n' + f
            # print result.value_counts()



if __name__ == "__main__":

    # original_data = pandas.read_csv('MDP_training_data.csv')
    od = pd.read_csv('MDP_Original_data.csv')
    headers = list(od.columns.values)
    staticHeader, allFeatures = headers[:6], headers[6:]

    # discretization
    discretization(allFeatures, od)

    IterNum, limit = 20, 8
    record = {}
    for i in range(IterNum):
        shuffle(allFeatures)
        unseen = allFeatures[:]
        selected_features = ['difficultProblemCountSolved']
        prior_ECR = induce_policy_MDP2(od, selected_features)
        print 'iteration ' + str(i)
        while unseen:
            # random add one
            random_f = unseen.pop()
            if random_f in selected_features:
                continue
            print 'checking random feature:\t' + random_f
            selected_features += [random_f]
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
                tmp_record = []
                for j, sf in enumerate(selected_features):
                    tmp_removed_features = selected_features[0:j] + selected_features[j+1:]
                    tmp_ECR = induce_policy_MDP2(od, tmp_removed_features)
                    tmp_record.append(tmp_ECR)
                    # found the one to abandon
                    if tmp_ECR > cur_ECR:
                        print 'remove '  + sf
                        selected_features = tmp_removed_features
                        cur_ECR = tmp_ECR
                        break
                # can't find one to abandon to improve the ECR
                if j >= limit:
                    # strategy 1: break
                    # print 'can\'t find one to abandon'
                    # break
                    # strategy 2 is to keep the highest ECR
                    h = tmp_record.index(max(tmp_record))
                    print 'abandon feature:\t' + selected_features[h]
                    selected_features = selected_features[0:h] + selected_features[h+1:]
            print 'add ' + selected_features[-1]
            prior_ECR = cur_ECR
        record[', '.join(selected_features)] = prior_ECR
        print '\n-------------------- Binary -------------------------'
        print 'ECR for this iteration: ' + str(prior_ECR)
        print 'Selected features:\n' + str(selected_features)
        print '---------------------------------------------\n\n'

    print 'done'





    selected_features = ['Level', 'probDiff']
    ECR_value = induce_policy_MDP2(od, selected_features)
