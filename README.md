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
