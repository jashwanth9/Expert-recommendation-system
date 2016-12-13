# Expert-recommendation-system
Recommender systems are now popular both commercially and in the research community, where many algorithms have been suggested for providing recommendations. These algorithms typically perform differently in various domains and tasks. In this repository we deal with different algorithms we used to build a recommendation system for suggesting how likely a user would answer a question using the byte cup data.

## General instructions 
All the code is in python and can be easily run by just cloning/downloading the repository. It already has all the features generted and the code for executing different algorithms can be found in the **Expert-recommendation-system/code** directory.

## Running the code
You need to have python 2.7 setup on your machine for running code on this repository. The code could be simply run with the following command **python [file_name]**

## Instructions for Running various Algorithms
* Linear and Logistic regression    
    Files associated :- linreg.py, logreg.py

* Collaborative Filtering 
    * User-based Collaborative Filtering    
        Files associated :- collabFiltering.py, collavFiltering_cross.py
    
    * Item-based Collaborative Filtering    
        Files associated :- collab_users_and_questions.py, collab_content_based_tags.py
    
* Content-Based Method
    * Content-Boosted Collaborative Filtering    
        Files associated :- contentBoostedCF.py, content_based_tfidf_reverse.py, contentBoosted_cross.py, content_based_with_tfidf.py, content_based.py, content_based_withgaussian.py

    * Content-Based Method With K Nearest Neighbors(KNN)    
        Files associated :- content_based_cold.py, content_tfidf_on_training.py      

* Hybrid Method      
    Files associated :- collab_content_based_tags.py
    
* Neural Networks    
    Required Libraries :- TensorFlow
    Files associated :- /code/neural net/nn_final.py and all the assoicated files are present
    
* XGBoost    
    Files associated :- tryxgb.py, xgb_nparray.py, xgb_nparray_hs1.py, xgb_submiss.py, xgb_submiss_hs1.py,
    
* Sparse Linear Method (SLIM)    
    Files associated :- predictSLIM.py, predictSLIMqu.py
    
* SVD++    
    This was run using various libraries like MyMediaLIte and librec using some of the feature files directly.
     
* MyMediaLite  
    Files associated :- This is a library and the commands to run the same are in the /MyMediaLite-3.10/nb.txt
     
* Matrix Factorization    
    Files associated :- MatrixFactorization.ipynb this is an iPython notebook you need ipython to run this.
