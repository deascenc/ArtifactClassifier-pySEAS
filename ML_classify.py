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

#ML packages
from sklearn.model_selection import train_test_split
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


if __name__ == '__main__':

    # Argument Parsing
    # -----------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', type = argparse.FileType('r'), 
        nargs = '+', required = False, 
        help = 'path to the .tsv file for classification')
    ap.add_argument('-uc', '--updateClass', default = None,
        nargs = '+', required = False, 
        help = 'directory to ica.hdf5, put the tsv path into the input arguemnt,\
        updates the ica.hdf5 based on classifier and .tsv file')
    ap.add_argument('-t','--train', action='store_true',
        help = 'train the classifier on the newest class_metric dataframe')
    ap.add_argument('-fc', '--force', action='store_true',
        help = 'force re-calculation')
    ap.add_argument('-cf', '--classifier', 
        default = './classifier.hdf5',
        type = argparse.FileType('r'), 
        nargs = '+', required = False, 
        help = 'path to the classifier.hdf5 file')
    ap.add_argument('-p', '--plot', action='store_true',
        help= 'Vizualize training outcome')
    args = vars(ap.parse_args())

    parent_dir = os.path.split(os.getcwd())[0]

    classifier = None
    assert()

    if args['classifier'] is None:
        for path in classifier_list:
            if os.path.exists(path):
                classifier = path
                print ('Classifier found at ' + classifier)
    else:
        classifier = args['classifier']
        if os.path.exists(classifier):
            print ('Classifier found at ' + classifier)
        else:
            classifier = None
    assert (classifier != None), 'Classifier file not found.'
    
    classMetricsPath = classifier[:-5] + '_metrics.tsv'
    thumbnailPath = classifier[:-5] + '_thumbnail.hdf5'
    confidencePath = classifier[:-5] + '_confidence.tsv'
    class_dir = os.path.dirname(classifier)
    update = False
    group = False

    if args['input'] != None:
        paths = [path.name for path in args['input']]
        print('Input found:')
        [print('\t'+path) for path in paths]

        for path in paths:
            print('Processing file:', path)

            if path.endswith('.hdf5'):
                 assert 

            elif path.endswith('.tsv'):
                try:
                    hdf5path = path.replace('_metrics.tsv', '.hdf5')
                    savepath = path

                print("No signal componets are found. Classifying components...")
                if path.endswith('.tsv'):
                    hdf5path = path.replace('_metrics.tsv', '.hdf5')
                    savepath = path

                elif path.endswith('.hdf5'):
                    hdf5path = path
                    data = pd.read_csv(savepath, sep = '\t', index_col='exp_ic')
                    update = True

                #put the dataframe on a scale 0 to 1 (expect age) and select only non-noise
                try:
                    datacopy = data.drop('age', axis =1).copy()
                except Exception as e:
                    print ('ERROR: ', e)
                    datacopy = data.copy()

                datacopy -= datacopy.min()
                datacopy /= datacopy.max()
                
                try:
                    datacopy['age'] = data['age']
                except Exception as e:
                    print ('ERROR: ', e)

                g = h5(classifier)
                print('Loading metric list to train the classifier from:', classifier)
                domain_vars = g.load('domain_keys')
                nodomain_vars = g.load('nodomain_keys') 

                print('\nTraining classifier with full dataset:')

                try:
                    main_data = pd.read_csv(classMetricsPath, sep = '\t', index_col='exp_ic')
                    print('\tImporting class metrics dataframe')
                    print(main_data.head())
                except Exception as e:
                    main_data = pd.DataFrame()
                    print('\tError importing class metrics dataFrame')
                    print('\tERROR : ', e)
                    assert not main_data.empty, 'Check path to metrics dataframe'


                
                domain = main_data.loc[main_data['threshold.area'] != 0].copy()
                nodomain = main_data.loc[main_data['threshold.area'] == 0].copy()

                y_train = domain['signal']
                X_train = domain[domain_vars]
                y2_train = nodomain['signal']
                X2_train = nodomain[nodomain_vars]
                
                rnd_clf = RandomForestClassifier(n_estimators = 40, max_features = 2)
                rnd_clf2 = RandomForestClassifier(n_estimators = 40, max_features = 2)

                rnd_clf.fit(X_train, y_train)
                rnd_clf2.fit(X2_train, y2_train)

                print('\tTraining classifier')
                domain = datacopy.loc[datacopy['threshold.area'] != 0].copy()
                nodomain = datacopy.loc[datacopy['threshold.area'] == 0].copy()

                domain = domain.fillna(value=0).loc[:, domain_vars]
                nodomain = nodomain.fillna(value=0).loc[:, nodomain_vars]

                print('\tPredicting classes')
                #run classifier
                data.loc[domain.index, 'm_signal'] = rnd_clf.predict(domain)
                data.loc[nodomain.index, 'm_signal'] = rnd_clf2.predict(nodomain)

                update = True

                # data['artifact'] = np.array(data['m_signal'] == 0).astype(int)
                # data.sort_index()
                # print('\tSaving artifact_component to ', hdf5path)

                # f = h5(hdf5path)
                # noise = f.load('noise_components')
                # artifact = noise.copy() * 0
                # artifact[noise==0] = data['artifact'].values.tolist()
                # f.save({'artifact_components':artifact})

            #Save file for future manipulations
            if update:
                if group:
                    if os.path.exists(savepath):
                        print('Updating exisiting file: ', savepath)
                        main_data = pd.read_csv(savepath, sep = '\t', index_col='exp_ic')
                        try:
                            main_data.drop(columns = 'anml')
                        except Exception as e:
                            print(e)
                        main_data = main_data.sort_index()
                        main_data = main_data.append(data, sort=True)
                        main_data = main_data.loc[~main_data.index.duplicated(keep='last')]
                        main_data = main_data.sort_index()

                        current_anml = 'nope'
                        j=0
                        for i in main_data.index.to_list():
                            if current_anml == i[:9]:
                                main_data.loc[i, 'anml'] = j
                            else:
                                current_anml = i[:9]
                                j+=1
                                main_data.loc[i, 'anml'] = j
                                
                        print('\nNumber of missing rows for the full dataset: {0} of {1}'.format(np.sum(np.isnan(main_data['temporal.min'])), len(main_data)))
                        main_data.to_csv(savepath, sep = '\t')
                    else:
                        print('Creating NEW file: ', savepath)
                        data.to_csv(savepath, sep = '\t')
                else:
                    print('\nSaving dataframe to:', savepath)
                    data.to_csv(savepath, sep = '\t')
            else:
                print('Metrics have already been made and artifact components have already been defined.')
                print('No changes were made to either file.  Check flags if you would like to update')
                print('either the experimental metrics or the class metrics.')
                print('Add the force flag if you would like to force the re-calculations.')

    if args['train']:
        if not args['updateDF']:
            try:
                main_data = pd.read_csv(classMetricsPath, sep = '\t', index_col='exp_ic')
                print('Importing dataframe\n------------------------------------')
                print(main_data.head())
            except Exception as e:
                main_data = pd.DataFrame()
                print('Error importing dataFrame')
                print('\t ERROR : ', e)
                assert not main_data.empty, 'Check path to metrics dataframe'


        
        #load classfier hdf5
        g = h5(classifier)

        #Current variables for training
        domain_vars = [
                     #spatial metrics
                     'spatial.min', 
                     'spatial.max', 
                     'threshold.area',
                     'region.extent', 
                     'region.minaxis', 'region.majaxis',
                     'region.eccentricity',
                     #temporal metrics
                     'freq.rangesz',
                     'temporal.max', 'temporal.std',  
                     'freq.range.low', 'freq.range.high']

        nodomain_vars =['spatial.min',
            'spatial.max',
            'spatial.COMdom.x',
            'spatial.COMdom.y',
            'freq.maxsnr',
            'spatial.std',
            'temporal.max',
            'temporal.autocorr',
            'freq.range.low',
            'temporal.std',
            'temporal.min']

        # for when the training keys get saved
        # new_vars = above_threshold_vars
        
        # for var in below_threshold_vars:
        #     if var in new_vars:
        #         pass
        #     else:
        #         new_vars.append(var)
        
        domain = main_data.loc[main_data['threshold.area'] != 0].copy()
        nodomain = main_data.loc[main_data['threshold.area'] == 0].copy()
        
        X_train, X_test, y_train, y_test = splitData(domain.loc[:,domain_vars].copy(), domain.loc[:,'signal'].copy())
        X2_train, X2_test, y2_train, y2_test = splitData(nodomain.loc[:,nodomain_vars].copy(), nodomain.loc[:,'signal'].copy())

        Xlen = len(domain)
        X2len = len(nodomain)

        Xfraction = Xlen / (Xlen + X2len)
        X2fraction = X2len / (Xlen + X2len) 

        classConfidence = pd.DataFrame(index=main_data.index)

        #X_train, X_test, y_train, y_test = train_test_split(X[new_vars], y, 
        #    test_size=0.3) # random_state=42)

        #headers
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

        logreg.fit(X2_train, y2_train)
        y2_pred = logreg.predict(X2_test)

        logreg_score2 = logreg.score(X2_test, y2_test)
        logreg_precision2 =  precision_score(y2_test, y2_pred)
        logreg_recall2 = recall_score(y2_test, y2_pred)
        logreg_signal2 = (np.sum(((y2_pred==1) & (y2_test==1))))/(np.sum((y2_test==1)))
        logreg_artifact2 = (np.sum(((y2_pred==0) & (y2_test==0))))/(np.sum(y2_test==0))

        logreg_score = (Xfraction * logreg_score) + (X2fraction * logreg_score2)
        logreg_precision = (Xfraction * logreg_precision) + (X2fraction * logreg_precision2)
        logreg_recall = (Xfraction * logreg_precision) + (X2fraction * logreg_precision2)
        logreg_signal = (Xfraction * logreg_precision) + (X2fraction * logreg_precision2)
        logreg_artifact = (Xfraction * logreg_precision) + (X2fraction * logreg_precision2)
    
        classConfidence.loc[X2_test.index, 'logreg_prob'] = logreg.predict_proba(X2_test)[:,1]
        classConfidence.loc[X2_train.index, 'logreg_prob'] = logreg.predict_proba(X2_train)[:,1]

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

        gnb_clf.fit(X2_train, y2_train)
        y2_pred = gnb_clf.predict(X2_test)

        gnb_score2 = gnb_clf.score(X2_test, y2_test)
        gnb_precision2 =  precision_score(y2_test, y2_pred)
        gnb_recall2 = recall_score(y2_test, y2_pred)
        gnb_signal2 = (np.sum(((y2_pred==1) & (y2_test==1))))/(np.sum((y2_test==1)))
        gnb_artifact2 = (np.sum(((y2_pred==0) & (y2_test==0))))/(np.sum(y2_test==0))

        gnb_score = (Xfraction * gnb_score) + (X2fraction * gnb_score2)
        gnb_precision = (Xfraction * gnb_precision) + (X2fraction * gnb_precision2)
        gnb_recall = (Xfraction * gnb_precision) + (X2fraction * gnb_precision2)
        gnb_signal = (Xfraction * gnb_precision) + (X2fraction * gnb_precision2)
        gnb_artifact = (Xfraction * gnb_precision) + (X2fraction * gnb_precision2)
    
        classConfidence.loc[X2_test.index, 'gnb_prob'] = gnb_clf.predict_proba(X2_test)[:,1]
        classConfidence.loc[X2_train.index, 'gnb_prob'] = gnb_clf.predict_proba(X2_train)[:,1]

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

        svm_clf.fit(X2_train, y2_train)
        y2_pred = svm_clf.predict(X2_test)

        svm_score2 = svm_clf.score(X2_test, y2_test)
        svm_precision2 =  precision_score(y2_test, y2_pred)
        svm_recall2 = recall_score(y2_test, y2_pred)
        svm_signal2 = (np.sum(((y2_pred==1) & (y2_test==1))))/(np.sum((y2_test==1)))
        svm_artifact2 = (np.sum(((y2_pred==0) & (y2_test==0))))/(np.sum(y2_test==0))

        svm_score = (Xfraction * svm_score) + (X2fraction * svm_score2)
        svm_precision = (Xfraction * svm_precision) + (X2fraction * svm_precision2)
        svm_recall = (Xfraction * svm_precision) + (X2fraction * svm_precision2)
        svm_signal = (Xfraction * svm_precision) + (X2fraction * svm_precision2)
        svm_artifact = (Xfraction * svm_precision) + (X2fraction * svm_precision2)
     
        classConfidence.loc[X2_test.index, 'SVC_prob'] = svm_clf.predict_proba(X2_test)[:,1]
        classConfidence.loc[X2_train.index, 'SVC_prob'] = svm_clf.predict_proba(X2_train)[:,1]

        #Random Forest Classifier
        # forest_scores = [0]*50
        # forest_art = [0]*50
        # for i in range(50):
        #     sys.stdout.write("\rRunnning {0}/50".format(i+1))
        #     sys.stdout.flush()
        #     rnd_clf = RandomForestClassifier(n_estimators = 40, max_features = 2)
        #     rnd_clf.fit(X_train, y_train)
        #     y_pred = rnd_clf.predict(X_test)

        #     rnd_score = rnd_clf.score(X_test, y_test)
        #     rnd_precision =  precision_score(y_test, y_pred)
        #     rnd_recall = recall_score(y_test, y_pred)
        #     rnd_signal = (np.sum(((y_pred==1) & (y_test==1))))/(np.sum((y_test==1)))
        #     rnd_artifact = (np.sum(((y_pred==0) & (y_test==0))))/(np.sum(y_test==0))
        #     classConfidence.loc[X_test.index, 'rnd_clf_prob'] = rnd_clf.predict_proba(X_test)[:,1]
        #     classConfidence.loc[X_train.index, 'rnd_clf_prob'] = rnd_clf.predict_proba(X_train)[:,1]

        #     forest_scores[i] = rnd_score
        #     forest_art[i] = rnd_artifact

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

        rnd_clf2 = RandomForestClassifier(n_estimators = 40, max_features = 2)
        rnd_clf2.fit(X2_train, y2_train)
        y2_pred = rnd_clf2.predict(X2_test)

        rnd_score2 = rnd_clf2.score(X2_test, y2_test)
        rnd_precision2 =  precision_score(y2_test, y2_pred)
        rnd_recall2 = recall_score(y2_test, y2_pred)
        rnd_signal2 = (np.sum(((y2_pred==1) & (y2_test==1))))/(np.sum((y2_test==1)))
        rnd_artifact2 = (np.sum(((y2_pred==0) & (y2_test==0))))/(np.sum(y2_test==0))

        rnd_score = (Xfraction * rnd_score) + (X2fraction * rnd_score2)
        rnd_precision = (Xfraction * rnd_precision) + (X2fraction * rnd_precision2)
        rnd_recall = (Xfraction * rnd_precision) + (X2fraction * rnd_precision2)
        rnd_signal = (Xfraction * rnd_precision) + (X2fraction * rnd_precision2)
        rnd_artifact = (Xfraction * rnd_precision) + (X2fraction * rnd_precision2)

        classConfidence.loc[X2_test.index, 'rnd_clf_prob'] = rnd_clf2.predict_proba(X2_test)[:,1]
        classConfidence.loc[X2_train.index, 'rnd_clf_prob'] = rnd_clf2.predict_proba(X2_train)[:,1]

        # print('\n\nForest Scores: ', np.round(forest_scores, decimals = 2))
        # print('\nForest Artifact Accuracy: ', np.round(forest_art, decimals = 2))
        # forest_scores = np.asarray(forest_scores)
        # forest_scores_avg = forest_scores.sum() / len(forest_scores)
        # forest_scores_std = forest_scores.std()

        # forest_art = np.asarray(forest_art)
        # forest_art_avg = forest_art.sum() / len(forest_art)
        # forest_art_std = forest_art.std()

        # print('\nScore Average: {0}\nScore Std Dev: {1}'.format(np.round(forest_scores_avg, decimals = 2), 
        #   np.round(forest_scores_std, decimals = 4)))
        # print('\nArtifact Average: {0}\nArtifact Std Dev: {1}'.format(np.round(forest_art_avg, decimals =2), 
        #   np.round(forest_art_avg, decimals =4)))        
        

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

        voting_clf.fit(X2_train, y2_train)
        y2_pred = voting_clf.predict(X2_test)

        voting_score2 = voting_clf.score(X2_test, y2_test)
        voting_precision2 =  precision_score(y2_test, y2_pred)
        voting_recall2 = recall_score(y2_test, y2_pred)
        voting_signal2 = (np.sum(((y2_pred==1) & (y2_test==1))))/(np.sum((y2_test==1)))
        voting_artifact2 = (np.sum(((y2_pred==0) & (y2_test==0))))/(np.sum(y2_test==0))

        voting_score = (Xfraction * voting_score) + (X2fraction * voting_score2)
        voting_precision = (Xfraction * voting_precision) + (X2fraction * voting_precision2)
        voting_recall = (Xfraction * voting_precision) + (X2fraction * voting_precision2)
        voting_signal = (Xfraction * voting_precision) + (X2fraction * voting_precision2)
        voting_artifact = (Xfraction * voting_precision) + (X2fraction * voting_precision2)
    
        classConfidence.loc[X2_test.index, 'voting_clf_prob'] = voting_clf.predict_proba(X2_test)[:,1]
        classConfidence.loc[X2_train.index, 'voting_clf_prob'] = voting_clf.predict_proba(X2_train)[:,1]        

        newData = pd.DataFrame()
        scores_df = pd.DataFrame(columns=['classifier', 'score', 'precision', 'recall', 'signal acc.', 'artifact acc.'])
        scores_df.loc[0] = ['LogisticRegression'] + [logreg_score] + [logreg_precision] + [logreg_recall] + [logreg_signal] + [logreg_artifact]
        # scores_df.loc[1] = ['GaussianNB'] + [gnb_score] + [gnb_precision] + [gnb_recall] + [gnb_signal] + [gnb_artifact]
        scores_df.loc[2] = ['SVM'] + [svm_score] + [svm_precision] + [svm_recall] + [svm_signal] + [svm_artifact]
        scores_df.loc[3] = ['RandomForest'] + [rnd_score] + [rnd_precision] + [rnd_recall] + [rnd_signal] + [rnd_artifact]
        scores_df.loc[4] = ['Voting'] + [voting_score] + [voting_precision] + [voting_recall] + [voting_signal] + [voting_artifact]
        scores_df.set_index('classifier', inplace=True)
        
        print('\n------------------------------------')
        print(scores_df.round(2))
        	
        classConfidence.to_csv(confidencePath, sep = '\t')

        g = h5(classifier)
        g.save({
            'rnd_clf_domain':rnd_clf, 
            'rnd_clf_nodomain': rnd_clf2, 
            'domain_keys': domain_vars, 
            'nodomain_keys': nodomain_vars})        

        if args['plot']:
            from sklearn.metrics import roc_auc_score
            from sklearn.metrics import roc_curve
            
            # hist_x = [1, 2, 3, 4]
            # hist_y = [1, 2, 3, 4]
            # plt.plot(hist_x, hist_y, 'r.')
            # plt.show()

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
            classConfidence.loc[X2_train.index,'predicted'] = rnd_clf2.predict(X2_train)
            classConfidence.loc[X2_test.index,'predicted'] = rnd_clf2.predict(X2_test)
            classConfidence = classConfidence.sort_index()
            # print(domain.index)
            # print(nodomain.index)
            classConfidence['x'] = np.arange(len(classConfidence))
            classConfidence.loc[domain.index, 'marked'] = domain.loc[domain.index, 'signal']
            classConfidence.loc[nodomain.index, 'marked'] = nodomain.loc[nodomain.index, 'signal']
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
                        plt.axvline(x=j, ymax = line  , color = 'r', alpha=0.5)
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
                estimators=[#('lr', logreg), 
                            #('gnb', gnb_clf), 
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
            plt.savefig('/home/ackmanlab/Documents/rnd_ROC.svg')
            plt.show()
