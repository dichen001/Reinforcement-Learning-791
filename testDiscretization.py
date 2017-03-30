import collections
import numpy as np
import pandas
import pandas.core.algorithms as algos
import mdptoolbox, mdptoolbox.example
import argparse
import operator

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
        for l in np.where(np.sum(A[act], axis=1) == 0)[0]:
            A[act][l][l] = 1
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
    print('\tECR after adding ' + selected_features[-1] + ':\t\t' + str(ECR_value))
    return ECR_value

def jitter(a_series, noise_reduction=1000000):
    return (np.random.random(len(a_series))*a_series.std()/noise_reduction)-(a_series.std()/(2*noise_reduction))


def discretization(features, data, thre=10, bins=2):
    for f in features:
        uniques = data[f].unique()
        l = len(uniques)
        if l > thre:
            # data[f] = pandas.cut(data[f], bins, labels=range(bins))
            # data[f] = pandas.qcut(data[f] + jitter(data[f]), 2, labels=range(2))
            ranges = algos.quantile(uniques, np.linspace(0, 1, bins+1))
            result = pandas.tools.tile._bins_to_cuts(data[f], ranges, include_lowest=True)
            # print result.value_counts()
            range_names = list(result.values.unique())
            data[f]= result.apply(lambda x : range_names.index(x))
            # print '\n' + f
            # print result.value_counts()


        # if type(data.loc[0, f]) == type(np.float64(1.11)):
        #     # discretize
        #     bins = min(10, len(data[f].unique()))
        #     # print f + ':\t' + str(bins)
        #     # od[f] = pct_rank_qcut(od[f], bins)
        #     data[f] = pandas.cut(data[f], bins, right=True, labels=range(bins))



if __name__ == "__main__":

    original_data = pandas.read_csv('MDP_Original_data.csv')
    # selected_features = ['Level', 'probDiff']
    # ECR_value = induce_policy_MDP2(original_data, selected_features)

    headers = list(original_data.columns.values)
    staticHeader, allFeatures = headers[:6], headers[6:]

    # ECR = 352.22255655
    # selected_features = ['difficultProblemCountSolved', 'cumul_F1Score', 'cumul_PreviousStepClickCountWE', 'cumul_TotalTime', 'ruleScoreDN', 'ruleScoreEQUIV', 'ruleScoreHS', 'ruleScoreTAUT']
    # ECR = 409.989749288
    selected_features = ['cumul_TotalPSTime', 'difficultProblemCountSolved', 'ruleScoreEQUIV', 'F1Score', 'ruleScoreIMPL', 'ruleScoreSIMP', 'cumul_NextStepClickCountWE', 'SeenWEinLevel']
    discretization(selected_features, original_data, thre=10, bins=2)
    # ECR = induce_policy_MDP2(original_data, selected_features)

    # training_data = original_data[staticHeader+selected_features].copy()
    # training_data.to_csv('MDP_training_data.csv')

    threshold, bins = range(2, 6, 1), range(2, 7, 1)
    performances = {}
    for t in threshold:
        for b in bins:
            test_data = original_data.copy()
            discretization(selected_features, test_data, thre=t, bins=b)
            ECR = induce_policy_MDP2(test_data, selected_features)
            performances['-'.join([str(t),str(b)])] = ECR
            print 'threshold:\t' + str(t) + '\tbins:\t' + str(b) + '\tECR:\t' + str(ECR)
    pandas.DataFrame.from_dict(performances).to_csv('discretization_results_compare.csv')
    print 'done!'







