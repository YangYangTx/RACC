# RACC
We provide 3 ways to reproduce the result:


Method 1: We provide a file named "submitbest.zip", which can generate the best result when submitted to the online platform. You can directly submit it for quick reproduction.


Method 2: We provide pre-trained models. You can execute "test.sh" for model inference, then execute python postCluster.py, and it will generate a "results.csv" file in the current directory. You can compress it into "results.zip" and submit it to the online platform for result reproduction.
step1: test.sh
step2: python postCluster.py


Method 3: We provide the steps to train the models from scratch. This method requires training multiple models and takes some time. The specific steps are as follows:


Step 1:  Data Processing
Modify line 77 of the testing code creatlist.py ，set rootDir to the training EO data path;
Run: python creatlist.py


"""
This will generate 16 txt files in the data directory：
1）EO 10-classification task: (v0-v5)-10 classification
train_task2_v1.txt
val_task2_v1.txt

2）Multi-modality task: (EO+SAR)-10 classification
train_task2_v0_fusion.txt
val_task2_v0_fusion.txt

3）4-classification task: -4 classification
train_task2_v0_4class.txt
val_task2_v0_4class.txt
"""



step2：Start Training
8 GPUs; please do not change the training parameters
Execute: bash train.sh


step3:Perform Testing
Modify line 47 of the testing code infer_single.py to set the test EO path: testPath to the EO_test directory;
Modify line 13 of the testing code postCluster.py to set the test EO path: rootPath to the EO_test directory;


Execute: bash test.sh. 
This will generate "results.csv".



step4: Submission
Submit results.csv online.
