# TUG-EDGCN

Encoder-Decoder Graph Convolutional Network for automatic Timed-Up-and-Go and Sit-to-Stand segmentation

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

We provide the collated skeleton data and labels in ```.../data/xxx/raw``` for the two Timed-Up-and-Go datasets. You may use ```.../data/xxx_split.py``` to split the data for cross validation.

The skeleton structure of the three datasets:  
<img width="500" alt="3" src="https://user-images.githubusercontent.com/115300137/194770587-09cedc31-703f-4bde-af79-bc0ce7dab287.PNG">  
The corresponding spatial graph can be found at ```.../utils/graph.py```.

* Training details can be found at ```train.py```.

## Introduction to Jitter and Shift score
Due to the page limit of the ICASSP conference, in this page we provide more details of the two new metrics introduced the paper.  

* Jitter score  
The jittering (over-segmentation/fragmentation) problem is the discontinuity in the predicted actions. The widely used Edit score can be used to measure the jittering problem but is only dependent on the number of jittered segments ```(I,II,III,VII)```. However, there are many other features of the jittered segments that can reflect the quality of prediction and the robustness of the network. Our Jitter score aims to take these features into consideration and evaluate the jittering problem in a more accurate way. The examples below shows how the Jitter score evaluate the jittering and action order problems in the predictions.

  The Jitter score is a weighted sum of an "action order penalty" and a "jitter penalty". In the following examples, "Jitter0.5" refers to the fact that the two penalties are equally weighted; "Jitter1" refers to only using the jitter penalty. For TUG and STS in our paper, there are few action order problems. Therefore, we apply Jitter1 to let the jitter score fully focus on the jittering problem. For some other action segmentation tasks, where action order may become a notable issue, an adjusted weight could be more helpful for the evaluation.
  
  The Jitter score takes the following features into consideration:  
  (1) Length of Jittered segments ```(III, IV)```: The longer the discontinuity in an action, the more penalty should be given.  
  (2) The distance of the jittered segments to the action boundary ```(III, V)```: A frame near the action boundaries can look similar with both the previous action and the following action. Therefore, a jittered segment which is very close to the action boundary is somewhat "forgivable" and should receive less penalty. In contrary, when a frame which is far from the action boundary is classified into the adjacent action or some other actions, a large problem of whether the network can understand an action may be raised. In this case, a larger penalty should be given.
  (3) The predicted action of the jittered segment ```(III, VI)```: Similar with (2), a frame classified into an action that is far from the nearby ground truth actions can indicate large problems with how well the network can understand actions. The penalty given to large distance and "irrelevant" action types can accumulate in the Jitter score.  
  
  In addition, ```(VII, VIII, IX, X)``` demostrate that the Jitter score can smartly distinguish which actions should be considered as "relevant" or "irrelevant" by an iterative algorithm. In ```(VII, VIII)```, the jittered segment is closer to the 3-4 action boundary. When the jittered segment is predicted as action 5 (or anything other than 3 or 4), the penalty is much larger. In ```(IX, X)```, the results are the opposite. 



