# IMPORT ALL MODULES
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import time as tm
import pandas as pd
import numpy as np

# CREATE DATASET
realdataobj = pd.read_csv("newsdataset/True.csv")
fakedataobj = pd.read_csv("newsdataset/Fake.csv")

# DATAFRAME AND VISUALIZE
print("DATASET LOOKS LIKE : ")
print(realdataobj.head(4))

# APPEND A TARGET COLUMN DENOTING FAKE OR REAL CLASS
realdataobj["Status"] = "Real"
fakedataobj["Status"] = "Fake"
joineddf = pd.concat([realdataobj, fakedataobj])
joinarray = joineddf.values
joindata = joinarray[:,0]
jointarget = joinarray[:,-1]
xtrain, xtest, ytrain, ytest = train_test_split(joindata, jointarget, random_state = 8)

# MEASURE OUR DATASET
print("\nFull Dataset contains the following distrubutions : ")
print("- ", len(realdataobj.index), " Real News")
print("- ", len(fakedataobj.index), " Fake News")

# VECTORISE COUNT OF WORDS IN TITLE COLUMN AND SCALE
limitsamples = 5000   # KNN ENCOUNTERS MEMORY ERROR FOR TOO LARGE DATASETS
xtrain, ytrain, xtest, ytest = xtrain[:limitsamples], ytrain[:limitsamples], xtest[:limitsamples], ytest[:limitsamples]
countvec = CountVectorizer().fit(xtrain)      # only include words appearing in more than 5 titles
xtrainvec = countvec.transform(xtrain)
print("\n",repr(xtrainvec))
words = countvec.get_feature_names()
print(words[2000:2020])
xtestvec = countvec.transform(xtest)

# IMPLEMENT PURE ML MODEL LOGISTIC REGRESSION CLASSIFIER 
score = cross_val_score(KNeighborsClassifier( n_jobs = -1), xtrainvec, ytrain, cv = 5, n_jobs = -1)
print("\nEach cross validated portion score array : ")
print(score)
print("Average CV Logistic regression training score : ", np.mean(score))

# + GRIDSEARCH AND CROSS VALIDATION FOR PARAMETERS
start = tm.time()
param_grid = {'n_neighbors': [20, 40, 60, 80], 'weights': ["uniform", "distance"], 'algorithm': ['ball_tree', 'kd_tree', 'brute']}
grid = GridSearchCV(KNeighborsClassifier(n_jobs = -1), param_grid, cv=5, n_jobs = -1)
grid.fit(xtrainvec, ytrain)
print("\nTHE BEST OF K NEAREST NEIGHBOURS : ")
print("Best cross-validation score: {:.2f}".format(grid.best_score_)) 
print("Best parameters: ", grid.best_params_)
print(grid.best_estimator_)

# ACCURACY ON TRAIN AND TEST SET
print("\nTraining Set Accuracy :")
print(round(grid.score(xtrainvec, ytrain), 4))
print("Test Set Accuracy : ")
print(round(grid.score(xtestvec, ytest), 4))

# PREDICT THE NEWS STATUS OF A SAMPLE NEWS TITLE
sometitle = "Donald Trump has hired Putin to be his wingman for destroying china"
sometitlearray = np.array([[sometitle]])
sometitlevec = countvec.transform(sometitlearray.ravel())
print("Test Article :", sometitle)
print("You likely enterred a", grid.predict(sometitlevec)[0], "article")
end = tm.time()
print("\nTotal time to run :", round(end-start, 3), "seconds")
