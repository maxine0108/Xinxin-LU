"""
    Python script to submit as a part of the project of ELTP 2020 course.
    
    This script serves as a template. Please use proper comments and meaningful variable names.
"""

"""
    Group Members:
        (1) Xinxin LU
        (2) Niccolo BORGIOLI
        (3) Yingqiang WANG
        etc.
"""

"""
    Import necessary packages
"""
import numpy as np
import pandas as pd
import os 
import spacy

#PreProcess
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

#To array and SVD
from sklearn.decomposition import TruncatedSVD

#Models
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier 


#Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

"""
    Your methods implementing the models.
    
    Each of your model should have a separate method. e.g. run_random_forest, run_decision_tree etc.
    
    Your method should:
        (1) create the proper instance of the model with the best hyperparameters you found
        (2) fit the model with a given training data
        (3) run the prediction on a given test data
        (4) return accuracy and F1 score
        
    Following is a sample method. Please note that the parameters given here are just examples.
"""


#HELPER FUNCTIONS
##############################################################################



#INFO: Vectorize text in training data and perform SVD (1000 features output)

def preprocessing(raw_column):
    
    #Import French
    spacy_nlp = spacy.load('fr_core_news_sm')
    spacy_nlp.max_length = 42010910
        
    #Deal with accents
    def normalize_accent(string):
        string = string.replace('á', 'a')
        string = string.replace('â', 'a')
        string = string.replace('é', 'e')
        string = string.replace('è', 'e')
        string = string.replace('ê', 'e')
        string = string.replace('ë', 'e')
        string = string.replace('î', 'i')
        string = string.replace('ï', 'i')
        string = string.replace('ö', 'o')
        string = string.replace('ô', 'o')
        string = string.replace('ò', 'o')
        string = string.replace('ó', 'o')
        string = string.replace('ù', 'u')
        string = string.replace('û', 'u')
        string = string.replace('ü', 'u')
        string = string.replace('ç', 'c')
        return string
        
    def raw_to_tokens(raw_string, spacy_nlp):
        # Write code for lower-casing
        string = raw_string.lower()
        
        # Write code to normalize the accents
        string = normalize_accent(string)
            
        # Write code to tokenize
        spacy_tokens = spacy_nlp(string)
            
        # Write code to remove punctuation tokens and create string tokens
        string_tokens = [token.orth_ for token in spacy_tokens if not token.is_punct]
        
        # Write code to join the tokens back into a single string
        clean_string = " ".join(string_tokens)
        
        return clean_string
    
    #Preprocess
    docs_clean = list()
    for i,seq in enumerate(raw_column):
      docs_clean.append(raw_to_tokens(seq, spacy_nlp)) #COMPUTATION
    
    
    #TFIDF
    #CountVectorizer
    vectorizer = CountVectorizer()
    X_sample = vectorizer.fit_transform(docs_clean) 
    #Features words
    features = vectorizer.get_feature_names()
    # Write code to create a TfidfVectorizer object
    tfidf = TfidfTransformer()
    # Write code to vectorize the sample text
    X_tfidf = tfidf.fit_transform(X_sample)

    #To array and SVD
    #instance
    svd = TruncatedSVD(n_components=1000, n_iter=5)
    #Fit-Transfrom
    X_red = pd.DataFrame(svd.fit_transform(X_tfidf))

    return X_red, X_tfidf

###########################################################################

#MODELS 

def model_XgBoost(X_train1, y_train1, X_test1, y_test1): #ACC: 0.77, #f1: 0.769
    
    clf = XGBClassifier(colsample_bylevel=0.6, colsample_bytree=0.6, gamma=2,
                        learning_rate=0.05, max_depth=7, min_child_weight=10,
                        n_jobs=-1, num_class=27, objective='multi:softmax',
                        subsample=0.6, n_estimators=600,
                        verbosity=1) 

    clf.fit(X_train1, y_train1) #COMPUTATION
    
    y_predicted = clf.predict(X_test1)
    xg_accuracy = accuracy_score(y_test1, y_predicted)
    xg_f1 = f1_score(y_test1, y_predicted, average='weighted')
    
    return xg_accuracy, xg_f1



def model_random_forest(X_train2, y_train2, X_test2, y_test2): #f1: 0.76

    clf = RandomForestClassifier(random_state=0)  
    clf.fit(X_train2, y_train2)

    y_predicted = clf.predict(X_test2)
    rf_accuracy = accuracy_score(y_test2, y_predicted)
    rf_f1 = f1_score(y_test2, y_predicted, average="weighted")

    return rf_accuracy, rf_f1



def model_bagging_DT(X_train2, y_train2, X_test2, y_test2): #F1: 0.72
    
    DT = DecisionTreeClassifier(min_samples_leaf = 1,min_samples_split = 2,
                                random_state = 0)
    bagging1 = BaggingClassifier(base_estimator=DT, n_estimators=10,
                                 max_samples=0.8, max_features=0.8).fit(X_train2, y_train2)
    
    y_predicted = bagging1.predict(X_test2)
    bag_DT_accuracy = accuracy_score(y_test2, y_predicted)
    bag_DT_f1 = f1_score(y_test2, y_predicted, average="weighted")
    
    return bag_DT_accuracy, bag_DT_f1
    


def model_bagging_RF(X_train2, y_train2, X_test2, y_test2): #F1: 0.74
    
    RFClassifier = RandomForestClassifier(random_state=0)
    bagging1 = BaggingClassifier(base_estimator=RFClassifier,
                                 max_samples=0.8, max_features=0.8).fit(X_train2, y_train2)
    
    y_predicted = bagging1.predict(X_test2)
    bag_RF_accuracy = accuracy_score(y_test2, y_predicted)
    bag_RF_f1 = f1_score(y_test2, y_predicted, average="weighted")
    
    return bag_RF_accuracy, bag_RF_f1



def model_grad_boost(X_train2, y_train2, X_test2, y_test2): #0.71
    
    GBClassifier = GradientBoostingClassifier(n_estimators = 30, max_leaf_nodes = 4,
                                              max_depth = 27,
                                              random_state = 0).fit(X_train2,y_train2)

    y_predicted = GBClassifier.predict(X_test2)
    gradBoost_accuracy = accuracy_score(y_test2, y_predicted)
    gradBoost_f1 = f1_score(y_test2, y_predicted, average="weighted")
    
    return gradBoost_accuracy, gradBoost_f1



def model_ada_boost(X_train2, y_train2, X_test2, y_test2): #0.71
    DT = DecisionTreeClassifier(random_state = 0)
    ABClassifier = AdaBoostClassifier(base_estimator=DT, n_estimators= 28,
                                      random_state=0).fit(X_train2,y_train2)

    y_predicted = ABClassifier.predict(X_test2)
    adaBoost_accuracy = accuracy_score(y_test2, y_predicted)
    adaBoost_f1 = f1_score(y_test2, y_predicted, average="weighted")
     
    return adaBoost_accuracy, adaBoost_f1



"""
   The main function should print all the accuracies and F1 scores for all the models.
   
   The names of the models should be sklearn classnames, e.g. DecisionTreeClassifier, RandomForestClassifier etc.
   
   Please make sure that your code is outputting the performances in proper format, because your script will be run automatically by a meta-script.
"""

if __name__ == "__main__":
    
    #INITIALIZE MATRICES
    #Load
    path = r'C:\Users\nicco\OneDrive\Bureau\EL\Rakuten\Data'  #CHOOSE PATH OF DATA HERE!
    os.chdir(path)
    
    X_temp = pd.read_csv('X_train_update.csv')
    y_train = pd.read_csv('Y_train_CVw08PX.csv', index_col=0)
    X_train = X_temp[['designation', 'productid']]
    X_train.head()
    
    X_train['designation'].shape #84 916
    raw_column = X_train['designation']
    y_train = y_train['prdtypecode']
    
    #Preprocess data
    X_red, X_tfidf = preprocessing(raw_column) #COMPUTATION
            
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_red, y_train, test_size=0.3)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_tfidf, y_train, test_size=0.3)
    
    # print('X_train1 shape',X_train1.shape)
    # print('X_test1 shape',X_test1.shape)
    # print('y_train1 shape',y_train1.shape)
    # print('y_test1 shape',y_test1.shape)
    
    # print('X_train2 shape',X_train2.shape)
    # print('X_test2 shape',X_test2.shape)
    # print('y_train2 shape',y_train2.shape)
    # print('y_test2 shape',y_test2.shape, '\n')
    
    
    #RUN MODELS
    xg_accuracy, xg_f1 = model_XgBoost(X_train1, y_train1, X_test1, y_test1)
    rf_accuracy, rf_f1 = model_random_forest(X_train2, y_train2, X_test2, y_test2)
    bag_DT_accuracy, bag_DT_f1 = model_bagging_DT(X_train2, y_train2, X_test2, y_test2)
    bag_RF_accuracy, bag_RF_f1 = model_bagging_RF(X_train2, y_train2, X_test2, y_test2)
    gradBoost_accuracy, gradBoost_f1 = model_grad_boost(X_train2, y_train2, X_test2, y_test2)
    adaBoost_accuracy, adaBoost_f1 = model_ada_boost(X_train2, y_train2, X_test2, y_test2)
    

    #PRINT
    #print the results
    print("XgBoost model: Acc, F1:", xg_accuracy, xg_f1, '\n')
    print("Random Forest model: Acc, F1:", rf_accuracy, rf_f1, '\n')
    print("Bagging of decision trees model: Acc, F1:", bag_DT_accuracy, bag_DT_f1, '\n')
    print("Bagging of Random forests model: Acc, F1:", bag_RF_accuracy, bag_RF_f1, '\n')
    print("Gradient Boosting model: Acc, F1:", gradBoost_accuracy, gradBoost_f1, '\n')
    print("AdaBoost model: Acc, F1:", adaBoost_accuracy, adaBoost_f1)


    

























