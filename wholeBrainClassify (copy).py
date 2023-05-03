#!/usr/bin/env python3
'''
Functions for classifying ICA domains

Authors: Sydney C. Weiser and Brian R. Mullen
Date: 2019-04-06
'''

import os 
import re
import sys

# import wholeBrain as wb
import numpy as np
import pandas as pd

sys.path.append('/home/feldheimlab/Documents/pySEAS/')

import seaborn as sns
import matplotlib 
import matplotlib.pyplot as plt
#from sklearn.externals import joblib

import scipy
from skimage.measure import label, regionprops
from multiprocessing import Process, Array, cpu_count, Manager
from sklearn.model_selection import StratifiedShuffleSplit

from scipy.signal import argrelextrema

try:
    from seas.hdf5manager import hdf5manager as h5
except Exception as e:
    print('Error importing hdf5manager')
    print('\t ERROR : ', e)

try:
    from seas.signalanalysis import *
except Exception as e:
    print('Error importing seas.signalanalysis')
    print('\t ERROR : ', e)

try:
    from seas.waveletAnalysis import waveletAnalysis as wave
except Exception as e:
    print('Error importing seas.waveletAnalysis')
    print('\t ERROR : ', e)
    

def sortNoise(timecourses=None, lag1=None, return_logpdf=False, method='KDE', verbose=False):
    '''
    Sorts timecourses into two clusters (signal and noise) based on 
    lag-1 autocorrelation.  
    Timecourses should be a np array of shape (n, t).

    Returns noise_components, a np array with 1 value for all noise 
    timecourses detected, as well as the cutoff value detected
    '''

    if method == 'KDE':
        from sklearn.neighbors import KernelDensity

        # calculate lag autocorrelations
        if lag1 is None:
            assert timecourses is not None, 'sortNoise requires either timecourses or lag1'
            lag1 = tca.lagNAutoCorr(timecourses, 1)

        # calculate minimum between min and max peaks
        kde_skl = KernelDensity(
            kernel='gaussian', bandwidth=0.05).fit(lag1[:, np.newaxis])
        x_grid = np.linspace(-0.2,1.2,1200)

        log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])

        maxima = argrelextrema(np.exp(log_pdf), np.greater)[0]
        if len(maxima) <= 1:
            if verbose: print('Only one cluster found')
            cutoff = 0
        else:
            cutoff_index = np.argmin(np.exp(log_pdf)[maxima[0]:maxima[-1]]) \
                + maxima[0]
            cutoff = x_grid[cutoff_index]
            if verbose: print('autocorr cutoff:', cutoff)

        noise_components = (lag1 < cutoff).astype('uint8')
    else:
        raise Exception('method: {0} is unknown!'.format(method))

    if return_logpdf:
        return noise_components, cutoff, log_pdf
    else:
        return noise_components, cutoff


def getPeakSeparation(log_pdf, x_grid=None):

    if x_grid is None:
        x_grid = np.linspace(-0.2,1.2,1200)
    
    maxima = argrelextrema(np.exp(log_pdf), np.greater)[0]

    if len(maxima) > 2:
        maxima = np.delete(maxima, np.argmin(np.exp(log_pdf)[maxima]))
    peak_separation = x_grid[maxima[-1]] - x_grid[maxima[0]]

    return peak_separation


def findContinBool(boolArray1D):
    nestDict = {}
    j, k = (-1, 0)
    usek = True
    for i in range(len(boolArray1D)):
        if boolArray1D[i] and not boolArray1D[i-1]:
            j += 1 
            if j >= 1:
                nestDict['region' + str(j-1)]['length'] = k
            k = 0
            nestDict['region' + str(j)] = {}
            nestDict['region' + str(j)]['freq.index'] = [] 
            nestDict['region' + str(j)]['freq.index'].append(i)
            k += 1
        if boolArray1D[i] and boolArray1D[i-1]:
            if i == 1:
                j += 1
                nestDict['region' + str(j)] = {}
                nestDict['region' + str(j)]['freq.index'] = []
                nestDict['region' + str(j)]['freq.index'].append(i-1)
                nestDict['region' + str(j)]['freq.index'].append(i)
            elif i == 0:
                pass
            else:
                k += 1
                nestDict['region' + str(j)]['freq.index'].append(i)

    if j != -1:
        nestDict['region' + str(j)]['length'] = k
    
    return nestDict


def findSig(wavelet):
    ratio = np.squeeze(wavelet.gws/wavelet.gws_sig)
    index = (ratio > 1)
    return ratio, index


def approxIntegrate(index, freq, power, sigcutoff): 
    #power
    diff = np.squeeze(power - sigcutoff) #integrate only significant area
    #freq
    if index[-1] != freq.shape[0]-1: # default to right sided estimation
        offset = [x+1 for x in index]
        binsz = freq[index] - freq[offset]
    else:
        offset = [x-1 for x in index]
        binsz = freq[offset] - freq[index]
        
    return np.sum(binsz * diff[index])


def temporalCharacterize(sigfreq, ratio, wavelet):
    for k in sigfreq.keys():
        sigfreq[k]['freq.maxsnr'] = np.nanmax(ratio[sigfreq[k]['freq.index']])
        sigfreq[k]['freq.maxsnr.freq'] = wavelet.flambda[np.nanargmax(ratio[sigfreq[k]['freq.index']])]
        sigfreq[k]['freq.avgsnr'] = np.nanmean(ratio[sigfreq[k]['freq.index']])
        sigfreq[k]['freq.range.low'] = wavelet.flambda[sigfreq[k]['freq.index'][-1]] 
        sigfreq[k]['freq.range.high'] = wavelet.flambda[sigfreq[k]['freq.index'][0]]
        if sigfreq[k]['freq.range.low'] and not sigfreq[k]['freq.range.high']:
                sigfreq[k]['freq.range.high'] = 5
        if sigfreq[k]['freq.range.high'] and not sigfreq[k]['freq.range.low']:
                sigfreq[k]['freq.range.low'] = 0                
        sigfreq[k]['freq.rangesz'] = (sigfreq[k]['freq.range.high'] - sigfreq[k]['freq.range.low'])
        sigfreq[k]['freq.integrate'] = approxIntegrate(index = sigfreq[k]['freq.index'], freq = wavelet.flambda, 
                                                  power = wavelet.gws, sigcutoff = wavelet.gws_sig)

    return sigfreq


def centerOfMass(eigenbrain, threshold=None, verbose=False, plot=False):
    eigen = eigenbrain.copy()
    if threshold is not None:
        eigen[eigen < threshold] = np.nan
    x, y = np.where(np.isnan(eigen)==False)

    #total sum
    totalmass = np.nansum(np.abs(eigen))
    #weighted sum
    sumrmx = 0
    sumrmy = 0
    for i in range(x.shape[0]):
        sumrmx += np.abs(eigenbrain[x[i],y[i]])*x[i]
        sumrmy += np.abs(eigenbrain[x[i],y[i]])*y[i]

#     plt.imshow(eigen)
#     plt.show()
    
    comx = sumrmx/totalmass
    comy = sumrmy/totalmass
   
    if verbose:
        print('xcom: ', comx)
        print('ycom: ', comy)

    if plot:
        plt.imshow(eigenbrain)
        plt.colorbar()
        plt.scatter(comy, comx, color='w', marker='*' )
        plt.show()

    return comx, comy


def xyProjectMax(eigenbrain, verbose=True, plot=True):
    xmean = np.nanmean(eigenbrain, axis = 1)
    xmax = np.nanargmax(xmean)
    
    ymean = np.nanmean(eigenbrain, axis = 0)
    ymax = np.nanargmax(ymean)

    if verbose:
        print('xmax: ', xmax)
        print('ymax: ', ymax)
        
    if plot:
        plt.plot(ymean)
        plt.plot(xmean)
        plt.scatter([xmax,ymax],[0,0], color = 'r')
        plt.show()
        
        
        plt.imshow(eigenbrain)
        plt.colorbar()
        plt.scatter(ymax, xmax, color='w', marker='*' )
        plt.show()

    return xmax, ymax

    
def positionMaxIntensity(eigenbrain, verbose=True, plot=True):
    amax = np.nanmax(eigenbrain)
    xamax, yamax = np.where(eigenbrain == amax)
    if verbose:
        print('x amax: ', xamax)
        print('y amax: ', yamax)
        
    if plot:
        plt.imshow(eigenbrain)
        plt.colorbar()
        plt.scatter(yamax, xamax, color='w', marker='*' )
        plt.show()
    return xamax, yamax
    
def spatialCharacterize(eigenbrain, threshold, verbose = False, plot = False):
    eigen = eigenbrain.copy()
#     eigen[eigen == np.nan] = 0
    # if np.nanmax(eigen) > threshold:
    eigen[eigen < threshold] = np.nan  
    x, y = np.where(np.isnan(eigen)==False)
    image = np.zeros_like(eigenbrain)
    image[x,y] = 1
    image = scipy.ndimage.median_filter(image, size=5)
    label_img = label(image)
    regions = regionprops(label_img, coordinates='rc')
    totalmass = np.nansum(np.abs(eigenbrain))
    
    domregion = {}
    
    for i, props in enumerate(regions):
        domregion['region' + str(i)] = {}
        regcoord = props.coords
        intensity = np.zeros_like(regcoord[:,0]).astype('float16')
        for j, coord in enumerate(regcoord):
            intensity[j] = eigenbrain[coord[0], coord[1]]
        
        if plot:
            plt.scatter(y, x, color = (0,0,0,0.05), marker='.')
            plt.scatter(regcoord[:,1], regcoord[:,0], c=intensity, cmap='jet')
            plt.gca().invert_yaxis()
            plt.show()
        
#         domregion['region' + str(i)]['pc_id']=pc
        domregion['region' + str(i)]['threshold.area']=props.area
        domregion['region' + str(i)]['threshold.perc']=props.area/np.sum(image)
        domregion['region' + str(i)]['mass.total']=totalmass
        domregion['region' + str(i)]['mass.region']=np.nansum(intensity)
        domregion['region' + str(i)]['mass.perc']=np.nansum(intensity)/totalmass
        domregion['region' + str(i)]['region.centroid.0']=props.centroid[0]
        domregion['region' + str(i)]['region.centroid.1']=props.centroid[1]
        domregion['region' + str(i)]['region.orient']=props.orientation
        domregion['region' + str(i)]['region.majaxis']=props.major_axis_length
        domregion['region' + str(i)]['region.minaxis']=props.minor_axis_length
        domregion['region' + str(i)]['region.extent']=props.extent
        domregion['region' + str(i)]['region.eccentricity']=props.eccentricity
        if props.minor_axis_length > 0:
            domregion['region' + str(i)]['region.majmin.ratio']=props.major_axis_length/props.minor_axis_length

        if verbose:
            print('Threshold area: ', props.area)
            print('Percent threshold area assessed: ', props.area/np.sum(image))
            print('\nTotalmass: ', totalmass)
            print('Areamass: ', np.nansum(intensity))
            print('Percent mass: ', np.nansum(intensity)/totalmass)
            print('\nCentroid: ', props.centroid)
            print('Orientation :', props.orientation)
            print('Major axis: ', props.major_axis_length)
            print('Minor axis: ', props.minor_axis_length)
            print('Extent: ',props.extent)
            print('Eccentricty: ',props.eccentricity)

    return domregion


def sortNestedDict(nestDict, sortkey = 'freq.rangesz'):
    sortDict = []
    for k in nestDict.keys(): sortDict.append(k)
#     sortDict = sorted(sortDict, key=lambda x: (sigfreq[x]['length']), reverse =True)
    sortDict = sorted(sortDict, key=lambda x: (nestDict[x][sortkey]), reverse =True)

    return sortDict
    

def _classMetrics(base, namespace, indexlist, eigenbrains, tcourses, roimask, threshold, fps, name):
    '''
    Child process for each CPU dedicated calculating metrics to determine classification.
    '''

    print("New process created: ", name)
    
    dataFrame = pd.DataFrame()
    dataFrame['exp_ic'] = [base[:-9] + '-' + '{}'.format(str(i).zfill(4)) for i in indexlist]
    dataFrame = dataFrame.set_index('exp_ic')

    print('\n', name, ': Calculating spatial metrics\n------------------------------------------------')
    dataFrame.loc[dataFrame.index.tolist(), 'spatial.std'] = np.nanstd(eigenbrains, axis = (1,2))
    dataFrame.loc[dataFrame.index.tolist(), 'spatial.avg'] = np.nanmean(eigenbrains, axis = (1,2))
    dataFrame.loc[dataFrame.index.tolist(), 'spatial.min'] = np.nanmin(eigenbrains, axis = (1,2))
    dataFrame.loc[dataFrame.index.tolist(), 'spatial.max'] = np.nanmax(eigenbrains, axis = (1,2))
    
    for i, pid in enumerate(dataFrame.index.tolist()):   
        if i%25 == 0:
            print(name, ': Working on {0} of {1} components'.format(i, eigenbrains.shape[0]))

        comxall, comyall = centerOfMass(eigenbrains[i])
        comxdom, comydom = centerOfMass(eigenbrains[i], threshold=threshold[i])

        #characterize the largest domain in the threshold
        k = spatialCharacterize(eigenbrains[i], threshold=threshold[i], plot = False)

        l = sortNestedDict(k, sortkey = 'mass.perc')
        dataFrame.at[pid,'spatial.n.domains'] = len(l)
        if len(l) > 0:
            for key, val in k[l[0]].items():
                dataFrame.at[pid, key] = val
                
        dataFrame.at[pid,'spatial.COMall.x'] = comxall      
        dataFrame.at[pid,'spatial.COMall.y'] = comyall        
        dataFrame.at[pid,'spatial.COMdom.x'] = comxdom      
        dataFrame.at[pid,'spatial.COMdom.y'] = comydom

    print('\n', name, ': Calculating temporal metrics\n------------------------------------------------')
    dataFrame.loc[dataFrame.index.tolist(),'temporal.autocorr'] = tca.lagNAutoCorr(tcourses, 1)
    dataFrame.loc[dataFrame.index.tolist(),'temporal.min'] = np.nanmin(tcourses, axis = 1)
    dataFrame.loc[dataFrame.index.tolist(),'temporal.max'] = np.nanmax(tcourses, axis = 1)
    dataFrame.loc[dataFrame.index.tolist(),'temporal.std'] = np.nanstd(tcourses, axis = 1)

    for i, pid in enumerate(dataFrame.index.tolist()):
        if i%25 == 0:
            print(name, ': Working on {0} of {1} components'.format(i, tcourses.shape[0]))
        w = wave(data = tcourses[i], fps=fps, mother = 'MORLET',
                 param = 4, siglvl = 0.99, verbose = False, plot=False)
        w.globalWaveletSpectrum()

        ratio, index = findSig(wavelet=w)
        m = findContinBool(index)
        m = temporalCharacterize(m, ratio, wavelet = w)
        n = sortNestedDict(m, sortkey = 'freq.rangesz')

        dataFrame.at[pid,'temporal.n.freq'] = len(n)
        if len(n) > 0:
            for key, val in m[n[0]].items():
                if key == 'freq.index':
                    pass
                else:
                    dataFrame.at[pid, key] = val

    try:
        namespace.df = namespace.df.join(dataFrame)
    except:
        namespace.df = namespace.df.fillna(dataFrame)
        
    print('\nProcess finished: ', name)

        
def batchClassify(base, dataFrame, eigenbrains, tcourses, roimask, threshold, notnoise_index, fps = 10, processes = 0):
    '''
    creates a number of processes to feed into however many CPUs dedicated to classify each component.  
    Recombines the data from each process into a single dataframe.
    '''    
    
    total_time = time.time()
    dt = time.time()

    if processes == 0: # if the user didn't specify the number of processes, use what is likely the best number
        wanted_processes = cpu_count()
        if wanted_processes > 12: wanted_processes = 12 # Max out at 8 CPUs
    else:
        wanted_processes = processes

    processes = []
    print("Creating " + str(wanted_processes) + " process(es)...")
    
    #put the dataframe in the multiprocessing namespace
    namespace = Manager().Namespace()
    namespace.df = dataFrame
    
    # the index map is used to distribute the brain data among the processes such that no process does much more work than any other
    index_map = []
    for i in range(wanted_processes):
        index_map.extend(list(np.arange(i,len(notnoise_index),wanted_processes)))
    index_map = np.asarray(index_map)
    
    dataPer = int(len(notnoise_index)/ wanted_processes) # number of rows of data per thread
    lower = 0
    upper = 0
    for i in range(wanted_processes): # create all processes
        name = "Process " + str(i+1)

        if i == wanted_processes-1: # for the last thread, give it any leftover rows of data, for example 23 rows, 5 processes, this process will do #17-23
            indices = index_map[upper:]
            p = Process(target=_classMetrics, args=(base, namespace, indices, eigenbrains[indices], tcourses[indices], 
                                                    roimask, threshold[indices], fps, name))
        else: # otherwise just divide up the rows into each process normally
            lower = i*dataPer
            upper = (i+1)*dataPer
            indices = index_map[lower:upper]
            p = Process(target=_classMetrics, args=(base, namespace, indices, eigenbrains[indices], tcourses[indices], 
                                                    roimask, threshold[indices], fps, name))
        p.start()
        processes.append(p)
    
    for i, p in enumerate(processes): # insure that all processes have finished
        p.join()
    print("All processes done")
        
    print('Time to complete process: {0:.0f} mins {1:.0f} secs'.format(np.floor((time.time() - total_time)/60), np.floor((time.time() - total_time)%60)))

    return namespace.df


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
    import pandas as pd
    from wholeBrainPCA import rebuildEigenbrain

    # Argument Parsing
    # -----------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', type = argparse.FileType('r'), 
        nargs = '+', required = False, 
        help = 'path to the processed ica file(s)')
    ap.add_argument('-f', '--fps', default = 10, required = False,
        help = 'frames per second from recordings')
    ap.add_argument('-g', '--group', nargs='+',
        help = 'metadata to group the experiments by.  options: age, drug, genotype')
    ap.add_argument('-pr', '--process', default = 0, required = False, type=int,
        help = 'Number of CPU dedicated to processing; 0 will max out the number of CPU')
    ap.add_argument('-n', '--noise_only', action='store_true',
        help = 'deterime noise components and save in hdf5')
    ap.add_argument('-t','--train', action='store_true',
        help = 'train the classifier on the newest class_metric dataframe')
    ap.add_argument('-ud', '--updateDF', action='store_true',
        help = 'update full classifier dataframe')
    ap.add_argument('-uc', '--updateClass', action='store_true',
        help = 'update class in experimental dataframe, input ica.hdf5 and ensure metrics.tsv is in the same folder')
    ap.add_argument('-fc', '--force', action='store_true',
        help = 'force re-calculation')
    ap.add_argument('-cm', '--classmetrics', type = argparse.FileType('r'), 
        nargs = '+', required = False, 
        help = 'path to the class_metrics.tsv file')
    ap.add_argument('-cf', '--classifier', type = argparse.FileType('r'), 
        nargs = '+', required = False, 
        help = 'path to the classifier.hdf5 file')
    ap.add_argument('-p', '--plot', action='store_true',
        help= 'Vizualize training outcome')
    ap.add_argument('-st', '--save_thumbnail', action='store_true',
        help= 'Saves IC spatial thumbnail to a thumbnail file in the classifier folder')
    args = vars(ap.parse_args())

    parent_dir = os.path.split(os.getcwd())[0]

    classifier_list = [ '/home/brian/Documents/data/Classifier/p21_classifier.hdf5',
                    '/Users/desiderioascencio/Downloads/p21_classifier.hdf5',
                    '/home/desi/Downloads/classifier/p21_classifier.hdf5',
                    '/hb/scratch/ackmanlab/classifier/p21_classifier.hdf5',
                    '/hb/groups/ackmanlab/classifier/p21_classifier.hdf5',
                    '/home/ackmanadmin/Documents/classifier/p21_classifier.hdf5',
                    os.path.join(parent_dir, 'classifier', 'p21_classifier.hdf5')]
    
    classMetricsPath = None
    classifier = None
    
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
                assert path.endswith('_ica.hdf5') | path.endswith('_ica_reduced.hdf5'),\
                     "Path did not end in '_ica.hdf5'"

                print('\nLoading data to create classifier metrics\n------------------------------------------------')
                f = h5(path)

                savepath = path.replace('.hdf5', '_metrics.tsv')
                savepath = savepath.replace('_reduced', '')
                savepath = os.path.basename(savepath)
                if not os.path.isdir(class_dir + '/expMetrics'):
                    os.mkdir(class_dir + '/expMetrics')
                wd = class_dir + '/expMetrics/'

                base = os.path.basename(path)
                # savepath = class_dir + '/expMetrics/' + os.path.basename(savepath)
                try:
                    expmeta = f.load('expmeta')
                    groupstring = ''
                    if 'age' in args['group']:
                        groupstring += expmeta['meta']['anml']['age'].upper() + '_'

                    if 'genotype' in args['group']:
                        groupstring += expmeta['meta']['anml']['genotype'] + '_'
                    if 'drug' in args['group']:
                        groupstring += expmeta['meta']['drug']['Anesthetic'] + \
                                        expmeta['meta']['drug']['AnesthPercent'] + '_'
                    savepath = wd + groupstring \
                                    + 'IC_metrics.tsv'
                    group = True
                except Exception as e:
                    print (e)
                    savepath = path.replace('.hdf5', '_metrics.tsv')
                    savepath = savepath.replace('_reduced', '')
                    savepath = wd + os.path.basename(savepath)
                
                if ('noise_components' not in f.keys()):
                    print('Calculating Noise Components')
                    noise, cutoff = sortNoise(f.load('timecourses'))
                    f.save({'noise_components':noise, 'cutoff':cutoff})
                else:
                    #Load data from ica.hdf5 file 
                    noise = f.load('noise_components')

                if args['noise_only']:
                    continue

                if 'artifact_components' in f.keys():
                    artifact = f.load('artifact_components')
                    if np.sum(artifact) != 0:
                        comb = noise + artifact
                        artifact[comb == 2] = 0 #if it is id'd as noise and artifact, keep as noise 
                        signal = np.array(comb == 0).astype(int)
                    else:
                        signal = np.zeros_like(noise)
                else:
                    artifact = np.zeros_like(noise)
                    signal = np.zeros_like(noise)

                notnoise_index = np.where(noise==0)[0]

                if args['updateClass'] and np.sum(signal)!=0 and os.path.exists(savepath):
                    data = pd.read_csv(savepath, sep = '\t', index_col='exp_ic')
                    print('\nImporting dataframe\n------------------------------------')
                    print('Sum of each component BEFORE update:\n',  data[['artifact','signal']].sum())
                    data['artifact'] = artifact[notnoise_index]
                    data['signal'] = signal[notnoise_index]
                    print('\nSum of each component AFTER update:\n', data[['artifact','signal']].sum())
                                    #Save file for future manipulations
                    print('Saving dataframe to:', savepath)
                    data.to_csv(savepath, sep = '\t')
                    update = True

                if (os.path.exists(savepath) and args['force']) or (not os.path.exists(savepath)):
                    if args['force']:
                        print('Re-calculating the metrics.')
                    if args['updateClass']:
                        print('Unable to update experimental dataframe. Either no artifact components or no metrics.tsv found')
                        print('Continuing on to create dataframe')
                    flipped = f.load('flipped')
                    print('Loading eig_mix for tcourse metrics')
                    tcourse = f.load('eig_mix')
                    roimask = f.load('roimask')
                    eig_vec = f.load('eig_vec')
                    thresh = f.load('thresholds')
                    try:
                         meta = f.load('expmeta')
                    except Exception as e:
                        print('Unable to add age to the dataFrame')
                        print('\t ERROR : ', e)                   

                    #flipped the inverted timeseries and 
                    tcourse = (np.multiply(tcourse, flipped)).T
                    eig_vec = np.multiply(eig_vec, flipped)

                    #create the dataframe and set up indices
                    data = pd.DataFrame()

                    data['artifact'] = artifact[notnoise_index]
                    data['signal'] = signal[notnoise_index]
                    try:
                        data['age'] = np.ones(int(signal[notnoise_index].shape[0])) * int(re.findall(r'\d+',meta['meta']['anml']['age'])[0])
                    except Exception as e:
                        print('ERROR: ', e)

                    data['exp_ic'] = [base[:-9] + '-' + '{}'.format(str(i).zfill(4)) for i in notnoise_index]
                    data = data.set_index('exp_ic')

                    eigenbrains = rebuildEigenbrain(eig_vec, roimask=roimask, bulk=True)
                    del eig_vec

                    if args['save_thumbnail']:
                        #save thumbnail store
                        t = h5(thumbnailPath)
                        t.save({base[:-9]:eigenbrains})

                    data = batchClassify(base, data, eigenbrains, tcourse, roimask, thresh, 
                                              notnoise_index, fps = args['fps'], processes = args['process'])

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
                        update = True

            elif path.endswith('.tsv'):
                try:
                    hdf5path = path.replace('_metrics.tsv', '.hdf5')
                    savepath = path

                    print('Importing dataframe\n------------------------------------')
                    data = pd.read_csv(path, sep = '\t', index_col='exp_ic')
                    signal = data['signal']
                    # try:
                    print('Importing class distinctions from ICA file\n------------------------------------')
                    if os.path.exists(hdf5path):
                        f = h5(hdf5path)
                        f.print()
                        
                        if ('noise_components' not in f.keys()) | args['force']:
                            print('Calculating Noise Components')
                            noise, cutoff = sortNoise(f.load('timecourses'))
                            f.save({'noise_components':noise, 'cutoff':cutoff})
                        else:
                            #Load data from ica.hdf5 file 
                            noise = f.load('noise_components')

                        if 'artifact_components' in f.keys():
                            artifact = f.load('artifact_components')
                            if np.sum(artifact) != 0:
                                comb = noise + artifact
                                artifact[comb == 2] = 0 #if it is id'd as noise and artifact, keep as noise 
                                signal_ica = np.array(comb == 0).astype(int)
                            else:
                                signal_ica = np.zeros_like(noise)
                        else:
                            artifact = np.zeros_like(noise)
                            signal_ica = np.zeros_like(noise)

                            notnoise_index = (noise==0)

                            data['artifact'] = artifact[notnoise_index]
                            data['signal'] = np.array(data['artifact'] == 0).astype(int)
                    else:
                        print("Could not find .hdf5 file.  Unable to update signal/artifact components.")

                    # difference = np.where((signal == 0.0) & (signal_ica != 0.0))

                    # if len(difference[0]) == 1:
                    #     print("Found a total of {0} change".format(len(difference[0])))
                    
                    # elif len(difference[0]) > 1: 
                    #     print("Found a total of {0} changes".format(len(difference[0])))

                    # else:
                    #     print("No changes detected.")

                    #Save file for future manipulations
                except Exception as e:
                    print('Error importing dataFrame')
                    print('\t ERROR : ', e)
                    raise AssertionError('Could not import dataframe from csv file.')
            
            else:
                base = os.path.basename(path)
                print('\nFile type {} not understood.'.format(base))
                print('skipping ' + path + '\n')

            if np.sum(signal)==0:
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

                from sklearn.svm import SVC
                from sklearn.model_selection import StratifiedShuffleSplit
                from sklearn.linear_model import LogisticRegression
                from sklearn.ensemble import RandomForestClassifier, VotingClassifier
                
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

            if args['updateDF']:
                #open dataframe
                try:
                    main_data = pd.read_csv(classMetricsPath, sep = '\t', index_col='exp_ic')
                    print('Importing dataframe\n------------------------------------')
                    print(main_data.head())
                except Exception as e:
                    main_data = pd.DataFrame()
                    print('Error importing dataFrame. Creating new main dataframe')
                    print('\t ERROR : ', e)

                #put the dataframe on a scale 0 to 1 (expect age) and select only non-noise
                try:
                    datacopy = data.drop('age', axis =1).copy()
                except Exception as e:
                    print("ERROR: ", e)
                    datacopy = data.copy()
                datacopy -= datacopy.min()
                datacopy /= datacopy.max()
                try:
                    datacopy['age'] = data['age']
                except Exception as e:
                    print("ERROR: ", e)
               #update dataframe
                main_data = main_data.combine_first(datacopy.fillna(value=0))
                
                #save dataframe
                main_data.to_csv(classMetricsPath, sep = '\t')

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

        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        from sklearn.model_selection import StratifiedShuffleSplit
        from sklearn.naive_bayes import GaussianNB
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from sklearn.metrics import precision_score, recall_score
        
        #load classfier hdf5
        g = h5(classifier)

        #Current variables for training
        domain_vars = [
                     #spatial metrics
                     'spatial.min', 'spatial.max', 
                     'threshold.area', #'mass.perc','threshold.perc', 'mass.region', #choose 1 
                     #'spatial.COMdom.x', 'spatial.COMdom.y', #positional argument - do we need?
                     'region.extent', 
                     'region.minaxis', 'region.majaxis',
                     'region.eccentricity', #'region.majmin.ratio',  #choose 1
                     
                     #temporal metrics
                     'freq.rangesz', #'length', #choose 1
                     'temporal.max', 'temporal.std',  #'temporal.autocorr', 
                     'freq.range.low', 'freq.range.high']

        nodomain_vars =[
            'spatial.min',
            'spatial.max',
            'spatial.COMdom.x',
            'spatial.COMdom.y',
            'freq.maxsnr',
        #   'spatial.COMall.x',
        #   'freq.avgsnr',
            'spatial.std',
            'temporal.max',
            'temporal.autocorr',
            'freq.range.low',
            'temporal.std',
            'temporal.min'
        #   'signal_index',
        #   'spatial.COMall.y'
        #   'freq.integrate',
        #   'freq.range.high',
        #   'freq.rangesz',
        #   'spatial.avg'
        ]

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
                        plt.axvline(x=j, ymax = linemax, color = 'r', alpha=0.5)
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
