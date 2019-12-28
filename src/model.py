import time
from copy import deepcopy
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, confusion_matrix

def create_cv_model(model, x, y, ids, cv_fold, cost_mat):

    s = time.time()

    accuracy = []
    auroc = [] 
    logloss = []
    profit_loss = []

    i = 0
    test_df = []

    for train_idx, test_idx in cv_fold.split(x, y):
        
        ### Fit model and make predictions
        model.fit(x[train_idx], y[train_idx])
        y_proba = model.predict_proba(x[test_idx])[:,1]
        y_pred = y_proba.round()
        
        ### Save predictions
        i += 1
        cv_test = pd.DataFrame([ids[test_idx], y_proba,y_pred, y[test_idx]]).T
        cv_test['split'] = i
        test_df.append(cv_test)

        ### Calculate evaluation metric per fold
        accuracy.append(accuracy_score(y[test_idx], y_pred))
        auroc.append(roc_auc_score(y[test_idx], y_proba))
        logloss.append(log_loss(y[test_idx], y_proba))

        conf_mat = confusion_matrix(y[test_idx], y_pred)
        conf_score = np.sum(conf_mat * cost_mat)
        profit_loss.append(conf_score)

    ### Concat predictions
    test_df = pd.concat(test_df)
    test_df.columns = ['vm_id', 'proba', 'pred', 'truth', 'split']
    
    duration = round(time.time() - s)
    
    model_fit_all = deepcopy(model)
    model_fit_all.fit(x, y)
    
        
    model_dict = {'model' : model_fit_all,
                  'test_df':test_df,
                  'auroc': auroc, 
                  'accuracy': accuracy, 
                  'logloss': logloss,
                  'profit_loss': profit_loss,
                  'duration' : duration}  
    
    
    return model_dict
	

	
	
def shap_summary(model_fit, x, feat_cols):
    
    if str(type(model_fit)).find('RandomForest') > 1:
        explainer = shap.TreeExplainer(model_fit)
        shap_values = explainer.shap_values(x)

        return shap.summary_plot(shap_values[1], x, feature_names=feat_cols)
        
    else: 
        x_s = x[:100]
        explainer = shap.KernelExplainer(model_fit.predict_proba, x_s)
        shap_values = explainer.shap_values(x_s)
        
        fig = plt.figure()
        shap.summary_plot(shap_values[0], x_s, feature_names=feat_cols)
        return fig
	
	
	
def plot_feature_importance(model, x, y, feat_cols):   
    model.fit(x, y)
    importances = model.feature_importances_

    imp_df = pd.DataFrame([feat_cols, importances] , index = ['feature', 'imp']).T.sort_values('imp')


    plt.barh(range(imp_df.shape[0]), imp_df['imp'], color = 'b', align = 'center')
    plt.yticks(range(imp_df.shape[0]), imp_df['feature'].values)
    plt.xlabel('Relative Importance')
    plt.show()
