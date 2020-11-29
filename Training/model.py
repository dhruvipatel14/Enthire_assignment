from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
import pickle
import numpy as np

class Train_model:
    def __init__(self,x_train,y_train):
        self.xtrain = x_train
        self.ytrain = y_train

    def random_forest(self):
        rfc = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth' : [4,5,6,7,8],
            'criterion' :['gini', 'entropy'],
        }

        CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
        CV_rfc.fit(self.xtrain,self.ytrain)

        best_param = CV_rfc.best_params_
        best_param.update(random_state=42 , class_weight='balanced')
        # rfc1 = RandomForestClassifier(random_state=42, class_weight='balanced',
        #                               max_features='auto', n_estimators=200,
        #                               max_depth=4, criterion='gini')

        rf_classifier = RandomForestClassifier(**best_param)
        rf_model = rf_classifier.fit(self.xtrain,self.ytrain)
        # with open('rf_classifier.pkl','wb') as file:
        #     pickle.dumps(rf_model,file)
        pickle.dump(rf_model, open("rf_classifier.pkl", "wb"))

    def logistic_regression(self):
        param_grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"],
                "class_weight": ['balanced', 'None']}  
        lgr = LogisticRegression()
        lgr_cv = GridSearchCV(lgr, param_grid, cv=10)
        lgr_cv.fit(self.xtrain, self.ytrain)

        best_params = lgr_cv.best_params_

        logistic_reg = LogisticRegression(**best_params)
        lgr_model = logistic_reg.fit(self.xtrain, self.ytrain)

        pickle.dump(lgr_model, open("logistic.pkl", "wb"))
        # with open('logistic.pkl','wb') as file:
        #     pickle.dumps(lgr_model,file)

    def xgboost_classifier(self):
        xgb_clf = xgb.XGBClassifier()
        xgb_model = xgb_clf.fit(self.xtrain, self.ytrain)
        #
        # with open('xgboost.pkl','wb') as file:
        #     pickle.dumps(xgb_model,file)

        pickle.dump(xgb_model, open("xgboost.pkl", "wb"))
















