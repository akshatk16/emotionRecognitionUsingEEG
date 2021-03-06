import operator
import pandas as pd
from scipy.sparse.construct import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm,tree,metrics
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.ensemble import AdaBoostClassifier as ada
from sklearn.neural_network import MLPClassifier
import warnings
from operator import itemgetter
import random
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score


rand = 7
models = {
    "NN":MLPClassifier(hidden_layer_sizes=(100, 10), activation="tanh", solver="sgd", learning_rate="adaptive"),
    "SVM":svm.SVC(C=1,random_state=rand),
    "DT":tree.DecisionTreeClassifier(criterion="entropy",max_depth=12,min_samples_leaf=2,min_samples_split=5,random_state=rand),
    "GBDT":GBDT(learning_rate=0.1,max_depth=9,min_samples_leaf=60,min_samples_split=10,n_estimators=31,random_state=rand),
    "ADA":ada(LogisticRegression(penalty='l2',C=0.55,max_iter=1000),random_state=rand),
    "LR":LogisticRegression(penalty='l2',C=0.55,max_iter=1000),
    "KNN":KNeighborsClassifier(3),
    "SVC":SVC(kernel="linear", C=0.025),
    "SVC2":SVC(gamma=2, C=1),
    "GPC":GaussianProcessClassifier(1.0 * RBF(1.0)),
    "DTC":DecisionTreeClassifier(max_depth=5),
    "RFC":RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    "MLPC":MLPClassifier(alpha=1, max_iter=1000),
    "ADABC":AdaBoostClassifier(),
    "GNB":GaussianNB()
}
warnings.filterwarnings('ignore')


def main():
    print("\nStarting...")
    df=pd.read_csv("Features.csv")
    print("\n> Feartures fetched!")
    
    y=df.iloc[:,-3] # Valence
    # y=df.iloc[:,-2] # Arousal
    # y=df.iloc[:,-1] # Dominance

    # print(y)
    X=df.iloc[:,0:-3]
    # print(X)

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=rand)
    print("\n> Dataset split into test and train sets!")

    print("\n~~~Evaluating models~~~\n")

    totalModels = len(models)
    data = []
    ranks = [i for i in range(1, totalModels+1)]
    headers = ["Model", "% Accuracy", "precision", "recall", "fscore"]
    modelsEvaluated = 0
    for i in models:
        modelsEvaluated += 1
        print("\r%d%% Models Evaluated" %(modelsEvaluated * 100 / totalModels), end = "")
        # choosing a model
        model=models[i]

        # fitting the model
        model.fit(X_train, y_train)

        # testing the model
        prediction=model.predict(X_test)


        precision, recall, fscore, support = score(y_test, prediction)
        
        # calculating accuracy of the model
        data.append([i, round(metrics.accuracy_score(y_test, prediction)*100, 2), round(precision[0]*100, 2), round(recall[0]*100, 2), round(fscore[0]*100, 2)])

    
    # print formatting

    print("\n")
    print(pd.DataFrame(data, ranks, headers), end="\n\n")
    pd.DataFrame(data).to_csv('models.csv')

if __name__ =="__main__":
    main()