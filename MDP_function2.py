import collections
import numpy as np
import pandas
import mdptoolbox, mdptoolbox.example
import argparse
import operator
import pandas.core.algorithms as algos


def generate_MDP_input2(original_data, features):

    students_variables = ['student', 'priorTutorAction', 'reward']

    # generate distinct state based on feature
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
    # distinct_states didn't contain terminal state
    student_list = list(data['student'].unique())
    distinct_states = list()
    for student in student_list:
        student_data = data.loc[data['student'] == student,]
        # don't consider last row
        temp_states = list(student_data['state'])[0:-1]
        distinct_states = distinct_states + temp_states
    distinct_states = list(set(distinct_states))

    Ns = len(distinct_states)

    # we include terminal state
    start_states = np.zeros(Ns + 1)
    A = np.zeros((Nx, Ns+1, Ns+1))
    expectR = np.zeros((Nx, Ns+1, Ns+1))

    # update table values episode by episode
    # each episode is a student data set
    for student in student_list:
        student_data = data.loc[data['student'] == student,]
        row_list = student_data.index.tolist()

        # count the number of start state
        start_states[distinct_states.index(student_data.loc[row_list[0], 'state'])] += 1

        # count the number of transition among states without terminal state
        for i in range(1, (len(row_list)-1)):
            state1 = distinct_states.index(student_data.loc[row_list[i - 1], 'state'])
            state2 = distinct_states.index(student_data.loc[row_list[i], 'state'])
            act = student_data.loc[row_list[i], 'priorTutorAction']

            # count the state transition
            A[act, state1, state2] += 1
            expectR[act, state1, state2] += float(student_data.loc[row_list[i], 'reward'])

        # count the number of transition from state to terminal
        state1 = distinct_states.index(student_data.loc[row_list[-2], 'state'])
        act = student_data.loc[row_list[-1], 'priorTutorAction']
        A[act, state1, Ns] += 1
        expectR[act, state1, Ns] += float(student_data.loc[row_list[-1], 'reward'])

    # normalization
    start_states = start_states / np.sum(start_states)

    for act in range(Nx):
        A[act, Ns, Ns] = 1
        # generate expected reward
        with np.errstate(divide='ignore', invalid='ignore'):
            expectR[act] = np.divide(expectR[act], A[act])
            expectR[np.isnan(expectR)] = 0

        # each column will sum to 1 for each row, obtain the state transition table
        # some states only have either PS or WE transition to other state
        for l in np.where(np.sum(A[act], axis=1) == 0)[0]:
            A[act, l, l] = 1
            
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
    # output_policy(distinct_acts, distinct_states, vi)

    # evaluate policy using ECR
    ECR_value = calcuate_ECR(start_states, vi.V)
    print('ECR value: ' + str(ECR_value))
    return ECR_value


def discretization(features, data, bins=10):
    for f in features:
        uniques = data[f].unique()
        l = len(uniques)
        if l > bins:
            # data[f] = pandas.cut(data[f], bins, labels=range(bins))
            # data[f] = pandas.qcut(data[f] + jitter(data[f]), 2, labels=range(2))
            ranges = algos.quantile(uniques, np.linspace(0, 1, 3))
            result = pandas.tools.tile._bins_to_cuts(data[f], ranges, include_lowest=True)
            range_names = list(result.values.unique())
            data[f]= result.apply(lambda x : range_names.index(x))


if __name__ == "__main__":

    original_data = pandas.read_csv('MDP_Original_data2.csv')
    # selected_features = ['Level']
    # ECR_value = induce_policy_MDP2(original_data, selected_features)



    # get best ECR
    # selected_features = ['cumul_TotalPSTime', 'difficultProblemCountSolved', 'ruleScoreEQUIV', 'F1Score', 'ruleScoreIMPL', 'ruleScoreSIMP', 'cumul_NextStepClickCountWE', 'SeenWEinLevel']
    # Best_ECR_value = induce_policy_MDP2(original_data, selected_features)


    headers = list(original_data.columns.values)
    staticHeader, allFeatures = headers[:6], headers[6:]

    discretization(allFeatures, original_data)

    ## feature selection policy [exploration part]
    print 'Stage 1: Exploration'
    LIMIT = 8
    selected = []
    rankings = [0]*LIMIT
    sorted_x = [0]*LIMIT
    for i in range(LIMIT):
        print '\n************ selecting feature: ' + str(i+1) + ' ************'
        rankings[i] = {}
        for f in allFeatures:
            if f in selected:
                continue
            ECR = induce_policy_MDP2(original_data, selected + [f])
            rankings[i][f] = ECR
        sorted_x[i] = sorted(rankings[i].items(), key=operator.itemgetter(1))
        selected += [sorted_x[i][-1][0]]
        print 'Selected Features: ' + selected[-1]
        print 'current ECR: ' + str(sorted_x[i][-1][1])
    print '\n@@@@@@@@@@@ Selected Features @@@@@@@@@@@'
    print selected
    print 'Exploration part done！'

    # ## try features after 5 selected
    # LIMIT = 8
    # selected = ['probDiff', 'Level', 'SolvedPSInLevel', 'cumul_avgstepTimeWE', 'cumul_TotalWETime']
    # rankings = [0] * LIMIT
    # sorted_x = [0] * LIMIT
    # for i in range(5, LIMIT):
    #     print '\n************ selecting feature: ' + str(i + 1) + ' ************'
    #     rankings[i] = {}
    #     for f in allFeatures:
    #         if f in selected:
    #             continue
    #         ECR = induce_policy_MDP2(original_data, selected + [f])
    #         rankings[i][f] = ECR
    #     sorted_x[i] = sorted(rankings[i].items(), key=operator.itemgett er(1))
    #     selected += [sorted_x[i][-1][0]]
    #     print 'Selected Features: ' + selected[-1]
    #     print 'current ECR: ' + str(sorted_x[i][-1][1])
    # print '\n@@@@@@@@@@@ Selected Features @@@@@@@@@@@'
    # print selected
    # print 'done'


    # ## try to replace current combination with a better choice
    # count = 0
    # rankings = [] * 20
    # sorted_x = [0] * 20
    # currentECR = 136.410583032
    # selected = ['probDiff', 'Level', 'SolvedPSInLevel', 'cumul_avgstepTimeWE', 'cumul_TotalWETime', 'stepTimeDeviation', 'ruleScoreDN', 'cumul_F1Score']
    # print 'current ECR: ' + str(currentECR)
    # while True:
    #     print '\n************ Attempt to Remove from ' + str(8 - count) + ' features ************'
    #     post_ECRs = []
    #     for i, f in enumerate(selected):
    #         ECR = induce_policy_MDP2(original_data, selected[:i] + selected[i+1:])
    #         post_ECRs += [(f, ECR)]
    #         print 'ECR after removing \t[' + f + ']:\t\t' + str(ECR)
    #     post_ECRs.sort(key=lambda x:x[1])
    #     new_ECR = post_ECRs[-1][1]
    #     # try all ECRs after removing the feature with minimum effects
    #     print '\nchoose to remove \t[' + post_ECRs[-1][0] + '], the left ECR:\t' + str(new_ECR)
    #     print '\n************ testing feature round: ' + str(count + 1) + ' ************'
    #     if currentECR - new_ECR >= 10:
    #         break


    # policy, step 2 [exploitation part]
    # bestECR = 136.410583032
    # selected = ['probDiff', 'Level', 'SolvedPSInLevel', 'cumul_avgstepTimeWE', 'cumul_TotalWETime', 'stepTimeDeviation', 'ruleScoreDN', 'cumul_F1Score']
    # selected = ['ruleScoreEXP', 'Level', 'TotalTime', 'cumul_avgstepTimeWE', 'cumul_TotalWETime', 'cumul_TotalPSTime', 'stepTimeDeviation', 'difficultProblemCountSolved']


    print 'Stage 2: Exploitation'
    bestECR = induce_policy_MDP2(original_data, selected)
    # for each feature selected, try to remove from the begining and see if we could find a better one.
    for i, feature in enumerate(selected):
        print '\n************ Attempt to Remove the NO.' + str(i + 1) + ' feature: [' + feature + '] ************'
        tmp_selected = selected[:i] + selected[i+1:]
        rankings = {}
        for f in allFeatures:
            if f in tmp_selected:
                continue
            ECR = induce_policy_MDP2(original_data, tmp_selected + [f])
            rankings[f] = ECR
        sorted_x = sorted(rankings.items(), key=operator.itemgetter(1))
        highFeature = sorted_x[-1][0]
        highECR = sorted_x[-1][1]
        print 'The highest ECR after removing No.' + str(i) + ' feature: ' + str(highECR)
        if highECR > bestECR:
            print '---- Replace [' + feature + '] with [' + highFeature + '] ----'
            print '---- ECR gets improved from ' + str(bestECR) + ' to ' + str(highECR)
            bestECR = highECR
            selected[i] = highFeature
        else:
            print '==== Cannot find a better feature to replace the No.' + str(i) + ' feature'

    print '\n+++++++++++++++++++++++++++++++++++++++'
    print 'The current ECR is ' + str(bestECR)
    print 'The selected features are:'
    print selected
    print 'Exploitatioin part done!'





