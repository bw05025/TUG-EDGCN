# TUG-EDGCN

Encoder-Decoder Graph Convolutional Network for automatic Timed-Up-and-Go and Sit-to-Stand segmentation

* Python: 3.8  
* PyTorch: 1.12.1  
* Numpy: 1.21.5  

## Qualitative Results
* Timed-Up-and-Go  
<img width="500" alt="1" src="https://user-images.githubusercontent.com/115300137/194768960-a8ba4b1b-1fc9-418e-9515-9d59f43e7a54.PNG">

* Fine-grained Sit-to-Stand
<img width="500" alt="2" src="https://user-images.githubusercontent.com/115300137/194768972-75d6f061-f4e6-4a9b-a0c9-ca611f2c8f90.PNG">

## Data and Training
* The TST-TUG dataset is a public dataset from [Università Politecnica delle Marche](https://www.tlc.dii.univpm.it/research/processing-of-rgbd-signals-for-the-analysis-of-activity-daily-life/kinect-based-dataset-for-motion-analysis) and is labeled and collated by us for machine learning applications.
* The Asian-TUG dataset is a public dataset from [Nanyang Technological University](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/7VF22X).
* The STS dataset is protected by the Institutional Review Boards (IRB) of University of California, San Diego. Therefore, we can't upload it here.

We provide the collated skeleton data and labels in ```.../data/xxx/raw``` for the two Timed-Up-and-Go datasets. You may use ```.../data/xxx_split.py``` to split the data for cross validation.

The skeleton structure of the three datasets:  
<img width="500" alt="3" src="https://user-images.githubusercontent.com/115300137/194770587-09cedc31-703f-4bde-af79-bc0ce7dab287.PNG">  
The corresponding spatial graph can be found at ```.../utils/graph.py```

