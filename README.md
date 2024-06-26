# TUG-EDGCN

Encoder-Decoder Graph Convolutional Network for automatic Timed-Up-and-Go and Sit-to-Stand segmentation ([published on ICASSP 2023](https://ieeexplore.ieee.org/abstract/document/10095810)).

* Python: 3.8  
* PyTorch: 1.12.1  
* Numpy: 1.21.5  

## Qualitative Results
* Timed-Up-and-Go:
<img width="500" alt="1" src="https://user-images.githubusercontent.com/115300137/194768960-a8ba4b1b-1fc9-418e-9515-9d59f43e7a54.PNG">

* Fine-grained Sit-to-Stand:
<img width="500" alt="2" src="https://user-images.githubusercontent.com/115300137/194768972-75d6f061-f4e6-4a9b-a0c9-ca611f2c8f90.PNG">


## Data and Training
* The TST-TUG dataset is a public dataset from [Università Politecnica delle Marche](https://www.tlc.dii.univpm.it/research/processing-of-rgbd-signals-for-the-analysis-of-activity-daily-life/kinect-based-dataset-for-motion-analysis) and is labeled and collated by us for machine learning applications.
* The Asian-TUG dataset is a public dataset from [Nanyang Technological University](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/7VF22X).
* The STS dataset is protected by the Institutional Review Boards (IRB) of University of California, San Diego. Therefore, we can't upload it here.

We provide the collated skeleton data and labels in ```.../data/xxx_TUG/raw``` for the two Timed-Up-and-Go datasets. You may use ```.../data/xxx_split.py``` to split the data for cross validation.

The skeleton structures of the three datasets:  
<img width="500" alt="3" src="https://user-images.githubusercontent.com/115300137/194770587-09cedc31-703f-4bde-af79-bc0ce7dab287.PNG">  
The corresponding spatial graphs can be found at ```.../utils/graph.py```.

* Training details can be found at ```train.py```.

## Introduction to Jitter and Shift score
Due to the page limit of the ICASSP conference, in this page we provide more details of the two new metrics proposed in the paper. The code is available at ```.../utils/metrics.py```.

* Jitter score  
The jittering (over-segmentation/fragmentation) problem is the discontinuity in the predicted actions. The widely used Edit score can be used to measure the jittering problem but is only dependent on the number of jittered segments ```(I,II,III,VII)```. However, there are many other features of the jittered segments that can reflect the quality of predictions and the robustness of the networks. Our Jitter score aims to take these features into consideration and evaluate the jittering problem in a more accurate way. The examples below show how the Jitter score evaluates the jittering and action order problems of the predictions.

  The Jitter score is a weighted sum of an "action order penalty" and a "jitter penalty". In the following examples, "Jitter0.5" refers to the fact that the two penalties are equally weighted; "Jitter1" refers to only using the jitter penalty. For TUG and STS in our paper, there are few action order problems. Therefore, we apply Jitter1 to let the jitter score fully focus on the jittering problem. For some other action segmentation tasks, where the action order may become a notable issue, an adjusted weight could be more helpful for the evaluation.
  
  The Jitter score takes the following features into consideration:  
  
    (1) Length of Jittered segments ```(III, IV)```: The longer the discontinuity in an action, the more penalty should be given.  
    (2) The distance of the jittered segments to the action boundary ```(III, V)```: A frame near the action boundaries can be similar with both the previous action and the following action. Therefore, a jittered segment which is very close to the action boundary is somewhat "forgivable" and should receive less penalty. On the contrary, when a frame which is far away from the action boundary is misclassified, a large problem of whether the network can accurately understand an action is raised. In this case, a larger penalty should be given.  
    (3) The predicted action of the jittered segment ```(III, VI)```: A frame classified into an action that is far away from the nearby ground truth actions can indicate large problem of how well the network can understand actions. The penalty given to large distance and "irrelevant" action types can accumulate in the Jitter score.  
  
  In addition, ```VII, VIII, IX, X``` demonstrate that the Jitter score can smartly distinguish which actions should be considered as "relevant" or "irrelevant" by an iterative algorithm. In ```VII, VIII```, the jittered segment is closer to the 3-4 action boundary. When the jittered segment is predicted as action 5 (or anything other than 3 or 4), the penalty will be much larger. In ```IX, X```, the results are the opposite.  

  Example ```XI, XII, XIII, XIV``` demonstrate how the Jitter score handles more chaotic predictions and the action order problem. In ```XI```, although the prediction is awful, each ground truth segment can still find a correponding segment in the prediction in a correct order. Therefore, in Jitter0.5, ```XI``` can still receive the full 50% score from the action order penalty. In the following three cases, as the action order problem worsens, the score becomes lower and lower. In all four cases, as the jittering problem is terrible, nearly full jitter penalty is given, resulting in Jitter1 be approximately 0. The penalization curve can be controlled by the parameter μ (see the paper and the code) if we want to allow more or less jittering problem.  
<img width="924" alt="4" src="https://user-images.githubusercontent.com/115300137/194781473-07c6f678-7751-4fac-b1b0-121c746d2275.PNG">


* Shift score  
  Another important problem in segmentation is the temporal shift. This problem reflects how well the network can locate each action. In fact, we find this problem influences the overall segmentation accuracy and prediction time error a lot more obviously than the jittering problem. The popular F1 score can be used to evaluate the temporal shift problem. However, constrained by its fixed-threshold design, it has three main drawbacks: (1) The F1 score will ignore small temporal shift; (2) For each segment, the result is binary, which will significantly damage the evaluation accuracy and (3) the jittering problem will also impact the result, which will produce noise in the evaluation of the temporal shift problem.  
  
  The following examples demonstrate how the Shift score outperforms the F1 score in evaluating the temporal shift problem.  
  
  Observe the orange segment in example ```I, II, III, IV```, as the temporal shift becomes increasingly large, the F1 score remains the same. Only after a certain threshold is met (50% IoU for F1@50) in ```V```, the F1 score drops. In contrast, the Shift score can capture these temporal shifts and gives larger penalty for larger shift. For multiple more shifts in example ```V, VI, VII```, the F1 score again fails to capture these changes, while the Shift score keeps giving larger penalties according to the magnitude of temporal shifts.  
  
  Comparing ```I```, ```VIII``` and the ground truth, we can see that the temporal shift problems are not very large. However, in ```VIII``` there is a serious jittering problem. As a result, the F1 score is impacted by the jittered segments and becomes lower. The Shift score, on the other hand, is much less susceptible to these jittered segments and can still produce an accurate result for the temporal shift problem.  
  
  ```IX, X``` show more complex cases. We can see that the Shift score can still produce reasonable evaluation for the temporal shift problem. In ```IX```, the yellow segment has a major temporal shift problem, resulting in a main impact to the score. The other segments are reasonably good (disregarding the jitterings) so the overall score is not that bad. In ```X```, only the green segment contributes a little to the score, the prediction for the other segments are all excessively bad, thus resulting in a very low score.  
<img width="924" alt="5" src="https://user-images.githubusercontent.com/115300137/194784567-a97362ad-53e0-45ab-9d75-ce5827156fa9.PNG">

  
