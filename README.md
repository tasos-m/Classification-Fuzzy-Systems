# Computational Intelligence - Classification Assignment

In this assignment, fuzzy TSK models are built to fit nonlineaar, multivariable functions, in order to solve classification problem.

## Part 1
The first part of the assignment is about the training process and the evaluation of four models. 
>Dataset used: [Haberman's Survival Dataset](https://archive.ics.uci.edu/ml/datasets/haberman's+survival) 

During the experiment these 4 models were tested:
* **TSK Model 1:** Subtractive clustering at the training data (class independent method) & Range of influence of the cluster center = 0.2 → 23 rules
* **TSK Model 2:** Subtractive clustering at the training data (class independent method) & Range of influence of the cluster center = 0.8 →  3 rules
* **TSK Model 3:** Subtractive clustering at the training data for each class separately (class dependent method) & Range of influence of the cluster center = 0.2 → 49 rules
* **TSK Model 4:** Subtractive clustering at the training data for each class separately (class dependent method) & Range of influence of the cluster center = 0.8 → 4 rules

### Error matrices and Accuracy of the 4 models
#### First model
|  | **Actual C1** | **Actual C2** |    
| --- | --- | --- |                     
| **Predicted C1**	| 44 | 8	|         
| **Predicted C2**	| 8 | 1	|

OA = 73.77%

#### Second model
|  | **Actual C1** | **Actual C2** |    
| --- | --- | --- |                     
| **Predicted C1**	| 49 | 8	|         
| **Predicted C2**	| 3 | 1	|

OA = 81.97%

#### Third model
|  | **Actual C1** | **Actual C2** |    
| --- | --- | --- |                     
| **Predicted C1**	| 36 | 7	|         
| **Predicted C2**	| 16 | 2	|

OA = 62.3%

#### Fourth model
|  | **Actual C1** | **Actual C2** |    
| --- | --- | --- |                     
| **Predicted C1**	| 50 | 9	|         
| **Predicted C2**	| 2 | 0	|

OA = 81.97%

## Part 2
In the second part, a large dataset of 11500 samples and 179 feautures is used. Thus it is important to reduce the number of feautures we use, to avoid the curse of dimensionality and the rule explosion. So before we train any model, we apply the _relieff_ function of the MATLAB Toolkit that ranks the importance of predictors, in order to choose the most important ones. Then, we devide the input space using Subtractive Clustering Technique, that is defined by the parameter "Range of influence of the cluster center". So we apply a Grid Search and 5-Fold Cross Validation to find which pair of range [0.3 0.5 0.8] and number of feautures [3 5 8 10] has the best performance, based on the validation error. Finally, we train the best model using the best pair of parameters accoridng to the Grid Search, evaluate the performance and comment the results.It's noted that we use Subtractive clustering at the training data for each class separately (class dependent method).
>Dataset used: [Epileptic Seizure Recognition Dataset](https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data) 

### Error matrices and Accuracy of the final model
|  | **Actual C1** | **Actual C2** | **Actual C3** | **Actual C4** |  **Actual C5** |
| --- | --- | --- | --- | --- | --- |
| **Predicted C1**	| 69 | 64	| 69	| 74 | 81 |
| **Predicted C2**	| 36 | 42	| 40	| 43 | 41 |
| **Predicted C3** | 200 | 188	| 213	| 194 | 200 |
| **Predicted C4** | 139| 147 |	138	| 148 | 111 |
| **Predicted C5** | 12 | 11 |	15	| 15 | 9 |

OA = 20.91%

The performance of the final model is not good and the from the accuracy metric we can infer that performs like random guessing, but the goal of the assignment was to get familiar with the concept and not to build the perfect model.

### Split Scale
In both parts we use the split_scale.m function to split the data in training, validation and checking data and there is an option to normalize or standardize them.
