import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy

import skimage.io
import skimage.transform
import imageio

from  sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Read and Resize images
def read_transform_img(file, img_width, img_height, img_channels):
    img = skimage.io.imread('data/input/images/' + file)
    img = skimage.transform.resize(img, (img_width, img_height), 
                                   mode='reflect')
    return img[:,:,:img_channels]
	
	
# Show Top Categories
def top_categories(df, col):
    cat_df = pd.DataFrame(df[col].value_counts())
    cat_df['relative'] = (cat_df[col] / df.shape[0]).round(2)
    cat_df.columns = ['absolute', 'relative']
    
    return cat_df
	
# Show one image per class of category
def show_category_imgs(df, col):
    cat_list = df[col].unique()
    cat_cols = len(cat_list)
    f, ax = plt.subplots(nrows=1,ncols=cat_cols, figsize=(12,3))
    i=0

    for s in cat_list:
        file='data/input/images/' + df[df[col]==s].iloc[0]['file']
        im=imageio.imread(file)
        ax[i].imshow(im, resample=True)
        ax[i].set_title(s, fontsize=8)
        i+=1

    plt.suptitle(col)
    plt.tight_layout()
    plt.show()
	
# Show Loss and Accuracy of Train and Loss
def show_fit_performance(model_fit):
	plt.subplot(1, 2, 1)
	plt.plot(model_fit.history['acc'])
	plt.plot(model_fit.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')

	plt.subplot(1, 2, 2)
	plt.plot(model_fit.history['loss'])
	plt.plot(model_fit.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')

	plt.tight_layout()
	plt.show()

	
# Confusion Matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
						  rotation = 45):
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
    plt.xticks(tick_marks, classes, rotation=rotation)
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
	

# Classification Report
def pandas_classification_report(y_true, y_pred, labels):
    metrics_summary = precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    accs = cm2.diagonal().round(2)

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index)
    
    class_report_df.columns = labels
    class_report_df.loc['accuracy'] = accs
    
    class_report_df = class_report_df.T.round(2)
    class_report_df['support'] = class_report_df['support'].astype('int')
    class_report_df = class_report_df[['support', 'accuracy', 'precision', 'recall', 'f1-score']]

    return class_report_df
	
	
# Visualize Layer Kernels
def visualize_layer_kernels(img, conv_layer, title):
    """
    Displays how input sample image looks after convolution by each kernel
    :param img: Sample image array
    :param conv_layer: Layer of Conv2D type
    :param title: Text to display on the top 
    """
    # Extract kernels from given layer
    weights1 = conv_layer.get_weights()
    kernels = weights1[0]
    kernels_num = kernels.shape[3]
    
    # Each row contains 3 images: kernel, input image, output image
    f, ax = plt.subplots(kernels_num, 3, figsize=(7, kernels_num*2))

    for i in range(0, kernels_num):
        # Get kernel from the layer and draw it
        kernel=kernels[:,:,:3,i]
        ax[i][0].imshow((kernel * 255).astype(np.uint8), vmin=0, vmax=255)
        ax[i][0].set_title("Kernel %d" % i, fontsize = 9)
        
        # Get and draw sample image from test data
        ax[i][1].imshow((img * 255).astype(np.uint8), vmin=0, vmax=255)
        ax[i][1].set_title("Before", fontsize=8)
        
        # Filtered image - apply convolution
        img_filt = scipy.ndimage.filters.convolve(img, kernel)
        ax[i][2].imshow((img_filt * 255).astype(np.uint8), vmin=0, vmax=255)
        ax[i][2].set_title("After", fontsize=8)
        
    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show() 