#!/usr/bin/env python3
'''
Functions for machine learning to classify ICA domains

Authors: Brian R. Mullen and Desi Ascencio
Date: 2019-04-06
'''

import os 
import re
import sys

# import wholeBrain as wb
import numpy as np
import pandas as pd

sys.path.append('/home/feldheimlab/Documents/pySEAS/')
# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

#ML packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import precision_score, recall_score
try:
    from seas.hdf5manager import hdf5manager as h5
except Exception as e:
    print('Error importing hdf5manager')
    print('\t ERROR : ', e)

def splitData(dataFrame, signal, value_fill=0, n_splits=10, test_size=0.30):
        
        y = signal
        X = dataFrame.fillna(value=value_fill)

        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)#random_state=42)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        return X_train, X_test, y_train, y_test

if __name__ == '__main__':

    import argparse
    import time
    # Argument Parsing
    # -----------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input_tsv', type = argparse.FileType('r'), 
        nargs = '+', required = False, 
        help = 'path to the .tsv file for classification')
    ap.add_argument('-h5', '--input_hdf5', type = argparse.FileType('r'), 
        nargs = '+', required = False, 
        help = 'path to the .hdf5 file for updating artifact classifications')
    ap.add_argument('-uc', '--updateClass', action='store_true',
        help = 'updates the ica.hdf5 based on classifier model and metrics .tsv file\
        requires tsv and hdf5 inputs')
    ap.add_argument('-g', '--group_path', default = None,
        nargs = '+', required = False, 
        help = 'save path to a file that groups the experiment.  If used\
         on experiments that have already been characterized, this will force \
         the re-calculation of any already processed datafile')
    ap.add_argument('-t','--train', action='store_true',
        help = 'train the classifier on the newest class_metric dataframe')
    ap.add_argument('-fc', '--force', action='store_true',
        help = 'force re-calculation')
    ap.add_argument('-cf', '--classifier', default = './classifier.hdf5', 
        nargs = '+', required = False, 
        help = 'path to the classifier.hdf5 file')
    ap.add_argument('-p', '--plot', action='store_true',
        help= 'Vizualize training outcome')
    args = vars(ap.parse_args())
    


    classifier = args['classifier'][0]
    class_dir = os.path.dirname(classifier)
    assert os.path.exists(class_dir), 'Classifier directory does not exist: {}'.format(class_dir)
    confidencepath = classifier[:-5] + '_confidence.tsv'

    try:
        group = args['group_path'][0]
        assert os.path.exists(os.path.dirname(group)), 'Unknown directory for group save: {}'.format(group)
    except:
        group = None

    if args['updateClass']:
        hdf5path = args['input_hdf5'][0]
        assert os.path.exists(hdf5path), 'Could not find hdf5 file: {}'.format(hdf5path)

    try:
        g = h5(classifier)
        print('Loading metric list to train the classifier from:', classifier)
        domain_vars = g.load('domain_keys')
        rnd_clf = g.load('rnd_clf')
    except:
        assert args['train'], 'No classifier found. Please train a classifier.'
        domain_vars =['spatial.min', 'spatial.max', #spatial metrics
                      'region.minaxis', 'threshold.area', 'region.extent', #morphometrics
                      'threshold.perc', 'region.majaxis', 'region.majmin.ratio', 
                      'temporal.min', #temporal metric
                      'freq.rangesz'] #frequency metric

    if args['input_tsv'] != None:
        paths = [path.name for path in args['input_tsv']]
        print('Input found:')
        [print('\t'+path) for path in paths]

        p = -1
        for path in paths:
            print('Processing file:', path)
            if path.endswith('.tsv') & os.path.exists(path):
                p += 1
                print('Loading data: ', path)
                if p == 0:
                    main_data = pd.read_csv(path, sep = '\t', index_col='exp_ic')
                else:
                    data = pd.read_csv(path, sep = '\t', index_col='exp_ic')
                    main_data = pd.concat([main_data, data])         
            else:
                print('DATA NOT FOUND OR UNKNOWN FILE FORMAT: ', path)

        main_data = main_data.sort_index()
        main_data = main_data.loc[~main_data.index.duplicated(keep='last')]
        main_data = main_data.sort_index()
        print('\nNumber of missing rows for the full dataset: {0} of {1}'.format(np.sum(np.isnan(main_data['temporal.min'])), len(main_data)))
        
        if p == 0: 
            group_load = False

    if args['updateClass']:
        
        print("Updating the classification of components based on current classifier.")

        try:
            datacopy = main_data.drop('age', axis =1).copy()
        except Exception as e:
            print ('ERROR: ', e)
            datacopy = main_data.copy()

        scaler = StandardScaler()
        scaler.fit(datacopy.values)
        datacopy[:] = scaler.transform(datacopy.values)

        print('\nClassifying the full dataset:')
        
        X_train = datacopy[domain_vars].fillna(value=0)

        try:
            y_train = main_data['neural'].fillna(value=0)
            print('Loading signal data')

        except Exception as e:
            print ('ERROR: ', e)
            print('Loading signal data')
            y_train = main_data['signal'].fillna(value=0)
            print('ytrain loaddd')

        if np.sum(y_train) == 0:
            print('No classifications were found.')
            print('\tPredicting classes')
            new = True
        else:
            print('Previous classification found.')
            print('\tPredicting classes, saving as a new column')
            new = False

        #run classifier
        main_data['m_signal'] = rnd_clf.predict(X_train)

        if not new:
            accuracy = np.sum(main_data['m_signal']==main_data['signal'])/len(main_data)
            print('Accuracy comparing human to machine classification: {} %'.format(np.round(accuracy*100,2)))

        main_data['artifact'] = np.array(main_data['m_signal'] == 0).astype(int)
        print('\tSaving artifact_component to ', hdf5path)
        
        if os.path.exists(hdf5path):
            f = h5(hdf5path)
            noise = f.load('noise_components')
            notnoise_index = np.where(noise==0)[0]
            indices = [base[:-9] + '-' + '{}'.format(str(i).zfill(4)) for i in notnoise_index]
            artifact = noise.copy() * 0
            artifact[notnoise_index] = main_data.loc[indices, 'artifact'].values.tolist()
            f.save({'artifact_components': artifact})
        else:
            print('hdf5 file was not found or specified.')

        if group == None:
            if group_load:
                print('Multiple files loaded, unsure as to what to save the file as. If you desire this data to be saved, use the group_path argument')
            else:
                main_data.to_csv(path, sep = '\t')
        else:
            main_data.to_csv(group, sep = '\t')
        

    #Save file for future manipulations

    if args['train']:

        g = h5(classifier)

        try:
            datacopy = main_data.drop('age', axis =1).copy()
        except Exception as e:
            print ('ERROR: ', e)
            datacopy = main_data.copy()

        scaler = StandardScaler()
        scaler.fit(datacopy.values)
        datacopy[:] = scaler.transform(datacopy.values)
        datacopy = datacopy.fillna(value=0)

        try:
            X_train, X_test, y_train, y_test = splitData(datacopy.loc[:,domain_vars].copy(), main_data.loc[:,'neural'].fillna(value=0).copy())
        except: 
            X_train, X_test, y_train, y_test = splitData(datacopy.loc[:,domain_vars].copy(), main_data.loc[:,'signal'].fillna(value=0).copy())

        Xlen = len(main_data)

        classConfidence = pd.DataFrame(index=main_data.index)

        #Logistic Regression
        logreg = LogisticRegression(C  = 8.25, solver = 'newton-cg', max_iter = 17.25)
        
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        
        logreg_score = logreg.score(X_test, y_test)
        logreg_precision =  precision_score(y_test, y_pred)
        logreg_recall = recall_score(y_test, y_pred)
        logreg_signal = (np.sum(((y_pred==1) & (y_test==1))))/(np.sum((y_test==1)))
        logreg_artifact = (np.sum(((y_pred==0) & (y_test==0))))/(np.sum(y_test==0))

        classConfidence.loc[X_test.index, 'logreg_prob'] = logreg.predict_proba(X_test)[:,1]
        classConfidence.loc[X_train.index, 'logreg_prob'] = logreg.predict_proba(X_train)[:,1]

        #Gaussian Naive Bayes
        gnb_clf = GaussianNB(var_smoothing= 0.009)
        
        gnb_clf.fit(X_train, y_train)
        y_pred = gnb_clf.predict(X_test)

        gnb_score = gnb_clf.score(X_test, y_test)
        gnb_precision =  precision_score(y_test, y_pred)
        gnb_recall = recall_score(y_test, y_pred)
        gnb_signal = (np.sum(((y_pred==1) & (y_test==1))))/(np.sum((y_test==1)))
        gnb_artifact = (np.sum(((y_pred==0) & (y_test==0))))/(np.sum(y_test==0))

        classConfidence.loc[X_test.index, 'gnb_prob'] = gnb_clf.predict_proba(X_test)[:,1]
        classConfidence.loc[X_train.index, 'gnb_prob'] = gnb_clf.predict_proba(X_train)[:,1]

        # Support Vector Machine Gaussian Kernal
        svm_clf = SVC(kernel="rbf", gamma=0.7, C=8, probability= True, degree=.01)
        
        svm_clf.fit(X_train, y_train)
        y_pred = svm_clf.predict(X_test)

        svm_score = svm_clf.score(X_test, y_test)
        svm_precision =  precision_score(y_test, y_pred)
        svm_recall = recall_score(y_test, y_pred)
        svm_signal = (np.sum(((y_pred==1) & (y_test==1))))/(np.sum((y_test==1)))
        svm_artifact = (np.sum(((y_pred==0) & (y_test==0))))/(np.sum(y_test==0))

        classConfidence.loc[X_test.index, 'SVC_prob'] = svm_clf.predict_proba(X_test)[:,1]
        classConfidence.loc[X_train.index, 'SVC_prob'] = svm_clf.predict_proba(X_train)[:,1]

        #Random Forest Classifier
        rnd_clf = RandomForestClassifier(n_estimators = 40, max_features = 2)
        rnd_clf.fit(X_train, y_train)
        y_pred = rnd_clf.predict(X_test)

        rnd_score = rnd_clf.score(X_test, y_test)
        rnd_precision =  precision_score(y_test, y_pred)
        rnd_recall = recall_score(y_test, y_pred)
        rnd_signal = (np.sum(((y_pred==1) & (y_test==1))))/(np.sum((y_test==1)))
        rnd_artifact = (np.sum(((y_pred==0) & (y_test==0))))/(np.sum(y_test==0))
        
        classConfidence.loc[X_test.index, 'rnd_clf_prob'] = rnd_clf.predict_proba(X_test)[:,1]
        classConfidence.loc[X_train.index, 'rnd_clf_prob'] = rnd_clf.predict_proba(X_train)[:,1]

        #Voting Classifier
        voting_clf = VotingClassifier(
            estimators=[('lr', logreg), 
                        #('gnb', gnb_clf), 
                        ('rf', rnd_clf), 
                        ('svc', svm_clf)], voting='soft')
        voting_clf.fit(X_train, y_train)
        y_pred = voting_clf.predict(X_test)
        
        voting_score = voting_clf.score(X_test, y_test)
        voting_precision =  precision_score(y_test, y_pred)
        voting_recall = recall_score(y_test, y_pred)
        voting_signal = (np.sum(((y_pred==1) & (y_test==1))))/(np.sum((y_test==1)))
        voting_artifact = (np.sum(((y_pred==0) & (y_test==0))))/(np.sum(y_test==0))

        classConfidence.loc[X_test.index, 'voting_clf_prob'] = voting_clf.predict_proba(X_test)[:,1]
        classConfidence.loc[X_train.index, 'voting_clf_prob'] = voting_clf.predict_proba(X_train)[:,1]   

        scores_df = pd.DataFrame(columns=['classifier', 'score', 'precision', 'recall', 'signal acc.', 'artifact acc.'])
        scores_df.loc[0] = ['LogisticRegression'] + [logreg_score] + [logreg_precision] + [logreg_recall] + [logreg_signal] + [logreg_artifact]
        scores_df.loc[1] = ['GaussianNB'] + [gnb_score] + [gnb_precision] + [gnb_recall] + [gnb_signal] + [gnb_artifact]
        scores_df.loc[2] = ['SVM'] + [svm_score] + [svm_precision] + [svm_recall] + [svm_signal] + [svm_artifact]
        scores_df.loc[3] = ['RandomForest'] + [rnd_score] + [rnd_precision] + [rnd_recall] + [rnd_signal] + [rnd_artifact]
        scores_df.loc[4] = ['Voting'] + [voting_score] + [voting_precision] + [voting_recall] + [voting_signal] + [voting_artifact]
        scores_df.set_index('classifier', inplace=True)
        
        print('\n------------------------------------')
        print(scores_df.round(2))
        	
        classConfidence.to_csv(confidencepath, sep = '\t')

        g = h5(classifier)
        g.save({
            'rnd_clf':rnd_clf, 
            'domain_keys': domain_vars})        

        if args['plot']:
            from sklearn.metrics import roc_auc_score
            from sklearn.metrics import roc_curve
            
            classConfidence = 2*(classConfidence - 0.5)
            # classConfidence.to_csv(classifier [:-5] + '_confidence.csv')
            start_exp = []
            end_exp = []
            for j, i in enumerate(classConfidence.index.tolist()):
                IC = int(i[-4:])
                if IC == 0:
                    start_exp.append(j)

            for i in start_exp:
                if i != 0:
                    end_exp.append(i-1)
            end_exp.append(len(classConfidence))

            alphas = [0.25, 0.25, 0.25, 0.25, 1] 
            colors = sns.color_palette()
            clfs = ['logreg_prob', 'gnb_prob', 'SVC_prob', 'rnd_clf_prob' ,'voting_clf_prob']
            
            classConfidence.loc[X_train.index,'predicted'] = rnd_clf.predict(X_train)
            classConfidence.loc[X_test.index,'predicted'] = rnd_clf.predict(X_test)
            classConfidence = classConfidence.sort_index()

            classConfidence['x'] = np.arange(len(classConfidence))
            classConfidence.loc[main_data.index, 'marked'] = main_data.loc[main_data.index, 'signal']
            classConfidence['false'] = classConfidence['predicted'] - classConfidence['marked']

            for i, clf in enumerate(clfs):
                if i == 0:
                    ax1 = classConfidence.plot(kind = 'scatter', x = 'x', y= clf, label = clf,
                                        color = colors[i], alpha = alphas[i], figsize=(20,4), grid=False)
                else:
                    classConfidence.plot(kind = 'scatter', x = 'x', y= clf, label = clf,
                                         color = colors[i], alpha = alphas[i], grid=False, ax = ax1)
            plt.axhline(y = 0, color = 'k', linestyle = '--')
            plt.ylabel('artifact                                  signal')

            xlabel = []
            falsePos = True
            falseNeg = True
            linemax = 0.94

            for j, i in enumerate(classConfidence['false'].tolist()):
                if i < 0:
                    if falseNeg:
                        plt.axvline(x=j, ymax = linemax, color = 'r', alpha=0.5, label = 'False Negative')
                        falseNeg = False
                    else:
                        plt.axvline(x=j, ymax = linemax , color = 'r', alpha=0.5)
                elif i > 0:
                    if falsePos:
                        plt.axvline(x=j, ymax = linemax, color = 'b', alpha=0.5, label = 'False Positive')
                        falsePos = False
                    else:
                        plt.axvline(x=j, ymax = linemax, color = 'b', alpha=0.5)

            for j, i in enumerate(start_exp):
                if (j%2==0):
                    plt.axvspan(xmin=i, xmax =end_exp[j]+1, color='k', alpha=0.1)
                plt.text(i, 1.05, classConfidence.index[i])
            
            # plt.xticks(start_exp, xlabel)
            leg = plt.legend(title = 'Types of Classifier', bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
            leg.get_frame().set_linewidth(0.0)
            plt.xlabel('Num of Domains')
            plt.ylim([-1.05,1.15])           
            plt.show()

            plt.figure()

            logreg.fit(X_train, y_train)

            roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
            fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
            plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % roc_auc)

            gnb_clf.fit(X_train, y_train)

            roc_auc = roc_auc_score(y_test, gnb_clf.predict(X_test))
            fpr, tpr, thresholds = roc_curve(y_test, gnb_clf.predict_proba(X_test)[:,1])
            plt.plot(fpr, tpr, label='GaussianNB (area = %0.2f)' % roc_auc)

            svm_clf.fit(X_train, y_train)

            roc_auc = roc_auc_score(y_test, svm_clf.predict(X_test))
            fpr, tpr, thresholds = roc_curve(y_test, svm_clf.predict_proba(X_test)[:,1])
            plt.plot(fpr, tpr, label='SVC (area = %0.2f)' % roc_auc)

            rnd_clf.fit(X_train, y_train)

            roc_auc = roc_auc_score(y_test, rnd_clf.predict(X_test))
            fpr, tpr, thresholds = roc_curve(y_test, rnd_clf.predict_proba(X_test)[:,1])
            plt.plot(fpr, tpr, label='Random Forest Classifier (area = %0.2f)' % roc_auc)

            voting_clf = VotingClassifier(
                estimators=[('lr', logreg), 
                            ('gnb', gnb_clf), 
                            ('rf', rnd_clf), 
                            ('svc', svm_clf)], voting='soft')

            voting_clf.fit(X_train, y_train)

            roc_auc = roc_auc_score(y_test, voting_clf.predict(X_test))
            fpr, tpr, thresholds = roc_curve(y_test, voting_clf.predict_proba(X_test)[:,1])
            plt.plot(fpr, tpr, label='Voting Classifier (area = %0.2f)' % roc_auc)

            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([-0.025, 1.0])
            plt.ylim([0, 1.025])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic')
            plt.legend(loc="lower right")
            # plt.savefig('rnd_ROC.svg')
            plt.show()
