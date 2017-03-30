The best policy is scripted in `feature_selection_best_policy.py`
The best ECR we have achieved so far is  **409.989749288** for `MDP_Original_data.csv`, with the following features:
```
['cumul_TotalPSTime', 
'difficultProblemCountSolved', 
'ruleScoreEQUIV', 
'F1Score', 
'ruleScoreIMPL', 
'ruleScoreSIMP', 
'cumul_NextStepClickCountWE', 
'SeenWEinLevel']
```
The selected features are discretized into binary features and are saved in `MDP_training_data.csv`

Following the requirements, we run `python MDP_process.py -input MDP_training_data.csv` and the the output is save in 'output.png'

Besides, we also compired the effects on ECR for diffrent discritizioin policies. The results are saved and visualized in `discretization_results_comparison.xlsx`

