#================
#Labels
#================
label_0 = 'Class 0'
label_1 = 'Class 1'

#================
#Packages
#================
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc, log_loss, accuracy_score, confusion_matrix

#================
#Confusion matrix
#================
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.grid(False)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#================
#ROC Curve
#================
def plot_roc(truth, preds, title = ''):
    
    fpr, tpr, threshold = roc_curve(truth, preds)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title(title)

#================
#Histogram
#================
def plot_hist(truth, preds, label_0 = label_0, label_1 = label_1):
    
    plt.hist(preds[truth == 1], label = label_1 + ' (1)', color = 'b', bins = 50, alpha = 0.5)
    plt.hist(preds[truth == 0], label = label_0 + ' (0)', color = 'r', bins = 50, alpha = 0.5)
    plt.legend(loc = 'upper right')
    plt.title('Histogram Predictions')

#==========================================
#Classification Plot (4 Evaluation Plots)
#==========================================
def classification_plot(truth, preds, scores = None, label_0 = label_0, label_1 = label_1):
    plt.figure(figsize=(12,8))

    plt.subplot(2,2,1)
    plot_hist(truth, preds)


    plt.subplot(2,2,2)
    plot_roc(truth, preds, 'ROC Curve')


    plt.subplot(2,2,3)
    cnf_mat = confusion_matrix(truth, preds.round())
    plot_confusion_matrix(cnf_mat, classes = [label_0, label_1])
    plt.grid(False)

    if scores is not None:
        plt.subplot(2,2,4)
        plt.boxplot(scores)
        plt.title('Cross Validation: Profit/Loss')

    plt.tight_layout()
    plt.show()
