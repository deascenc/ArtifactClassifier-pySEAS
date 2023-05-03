#!/usr/bin/env python3
'''
Functions for creating metrics to classify ICA domains

Authors: Sydney C. Weiser and Brian R. Mullen
Date: 2019-04-06
'''
# import systems level
import os 
import re
import sys
sys.path.append('/home/feldheimlab/Documents/pySEAS/') # dir of cloned repo

# key data management/manipulation packages
import numpy as np
import pandas as pd

# packages for characterizing shapes
import scipy
from skimage.measure import label, regionprops

# packages for parallelizing processing
from multiprocessing import Process, Array, cpu_count, Manager

# SEAS packages
try:
    from seas.hdf5manager import hdf5manager as h5
except Exception as e:
    print('Error importing hdf5manager')
    print('\t ERROR : ', e)

try:
    from seas.ica import rebuild_eigenbrain
except Exception as e:
    print('Error importing seas.ica')
    print('\t ERROR : ', e)

try:
    from seas.waveletAnalysis import waveletAnalysis as wave
except Exception as e:
    print('Error importing seas.waveletAnalysis')
    print('\t ERROR : ', e)

try:
    from seas.signalanalysis import sort_noise, lag_n_autocorr
except Exception as e:
    print('Error importing seas.waveletAnalysis')
    print('\t ERROR : ', e)
    

def findContinBool(boolArray1D):
    '''
    Used in frequency identification and characterization of continuous 
    significant frequencies based on wavelet transform 

    Arguments:
        boolArray1D: 1d array composed of 0,1 values. 
            1 indicates significant, 0 indicates not sigificant

    Returns: 
        nested dictionary of the characteristics of the continuous boolean
        value from the 1D array
    '''

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

    '''
    Used to create power-noise significance ratio and identify which
    index pass threshold

    Arguments:
        wavelet: pass in the wavelet object after global wavelet spectrum (GWS)
            and its significance has been calculated

    Returns: 
        1d vectors of ratios across all frequencies
        1d vectors of indices that achieve significance
    '''

    ratio = np.squeeze(wavelet.gws/wavelet.gws_sig)
    index = (ratio > 1)
    return ratio, index


def approxIntegrate(index, freq, power, sigcutoff):
    '''
    Approxamates integration by summing bin size and power of wavelet spectra

    Arguments:
        index: 1d vectors of indices that achieve significance
        freq: 1d vectors of corresponding frequencies (these are not equadistant to eachother)
        power: 1d global wavelet spectrum (GWS)
        sigcutoff: 1d GWS significance

    Returns: 
        1d vectors of ratios across all frequencies
        1d vectors of indices that achieve significance
    '''

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
    '''
    Creates atemporalCharacterize nested characterization of each component

    Arguments:
        index: 1d vectors of indices that achieve significance
        freq: 1d vectors of corresponding frequencies (these are not equadistant to eachother)
        power: 1d global wavelet spectrum (GWS)
        sigcutoff: 1d GWS significance

    Returns: 
        1d vectors of ratios across all frequencies
        1d vectors of indices that achieve significance
    '''
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


def centerOfMass(eigenbrain, threshold=None, verbose=False):
    '''
    center of mass based on intensity of the spatial component

    Arguments:
        index: 1d vectors of indices that achieve significance
        freq: 1d vectors of corresponding frequencies (these are not equadistant to eachother)
        power: 1d global wavelet spectrum (GWS)
        sigcutoff: 1d GWS significance

    Returns: 
        1d vectors of ratios across all frequencies
        1d vectors of indices that achieve significance
    '''
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
    
    comx = sumrmx/totalmass
    comy = sumrmy/totalmass
   
    if verbose:
        print('xcom: ', comx)
        print('ycom: ', comy)

    return comx, comy

    
def spatialCharacterize(eigenbrain, threshold, verbose = False):
    '''
    Creates nested dictionary based a spatial characteristics

    Arguments:
        eigenbrain: Single spatial component already rebuild for characterization
        threshold: Corresponding threshold (created by seas.ica) for spatial analysis

    Returns: 
        nested dictionary of all regions identified through regionprops
    '''

    eigen = eigenbrain.copy()
    eigen[eigen < threshold] = np.nan
    x, y = np.where(np.isnan(eigen)==False)

    image = np.zeros_like(eigenbrain)
    image[x,y] = 1
    image = scipy.ndimage.median_filter(image, size=5)
    label_img = label(image)
    regions = regionprops(label_img)#, coordinates='rc')
    totalmass = np.nansum(np.abs(eigenbrain))
    
    domregion = {}
    
    for i, props in enumerate(regions):
        domregion['region' + str(i)] = {} 
        regcoord = props.coords
        intensity = np.zeros_like(regcoord[:,0]).astype('float16')
        for j, coord in enumerate(regcoord):
            intensity[j] = eigenbrain[coord[0], coord[1]]
        
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
    '''
    Sorts the nested dictionary based a defined characteristic

    Arguments:
        nestDict: nested dictionary that has all the characterized 
            for spatial metrics, 'mass.perc' is used as the sorting mechanism. 
                Largest area is returned
            for temporal metrics, 'freq.rangesz' is used as the sorting mechanism. 
                Most frequecy representation is returned
        sortkey: key by which dictionary is sorted based on that value 

    Returns: 
        sorted dictionary based on the sort key
    '''
    sortDict = []
    for k in nestDict.keys(): sortDict.append(k)
    sortDict = sorted(sortDict, key=lambda x: (nestDict[x][sortkey]), reverse =True)

    return sortDict


def batchClassify(base, dataFrame, eigenbrains, tcourses, roimask, threshold, notnoise_index, fps = 10, processes = 0):
    '''
    creates a number of processes to feed into however many CPUs dedicated to classify each component.  
    Recombines the data from each process into a single dataframe.
    
    Arguments:
        base: base name for each index
        dataFrame: dataframe where all characterizations are stored
        eigenbrains: Spatial components already rebuild for characterization
        tcourses: Temporal components for characterization
        roimask: roimask of the decompositions
        threshold: list of thresholds (created by seas.ica) for spatial analysis
        notnoise_index: index of all components that are not considered noise
        fps: frames per second, default 10 fps for our recording setup 
        processes: Number of CPUs to parralize characterization, default set to 0 which then querries the 
            computers capabilities

    Returns
        dataframe filled with characerizations of each component (spatial, morphometirc, temporal, frequency)

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


def _classMetrics(base, namespace, indexlist, eigenbrains, tcourses, roimask, threshold, fps, name):
    '''
    Child process for each CPU dedicated calculating metrics to determine classification.  
    See parent function: batchClassify
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
        k = spatialCharacterize(eigenbrains[i], threshold=threshold[i])

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
    dataFrame.loc[dataFrame.index.tolist(),'temporal.autocorr'] = lag_n_autocorr(tcourses, 1)
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


if __name__ == '__main__':

    import argparse
    import time

    # Argument Parsing
    # -----------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', type = argparse.FileType('r'), 
        nargs = '+', required = False, 
        help = 'path to the ica that needs characterization or tsv \
        file that needs class update')
    ap.add_argument('-f', '--fps', default = 10, required = False,
        help = 'frames per second from recordings')
    ap.add_argument('-g', '--group_path', default = None,
        nargs = '+', required = False, 
        help = 'save path to a file that groups the experiment.  If used\
         on experiments that have already been characterized, this will force \
         the re-calculation of any already processed datafile')
    ap.add_argument('-pr', '--process', default = 0, required = False, type=int,
        help = 'Number of CPU dedicated to processing; 0 will \
        max out the number of CPU')
    ap.add_argument('-uc', '--updateClass', default = None,
        nargs = '+', required = False, 
        help = 'directory to ica.hdf5, put the tsv path into the input arguemnt,\
        updates the tsv based on ica.hdf5 classification')
    ap.add_argument('-fc', '--force', action='store_true',
        help = 'force re-calculation if not grouped')
    args = vars(ap.parse_args())

    parent_dir = os.path.split(os.getcwd())[0]

    savepath = None    
    update = False
    try:
        group = args['group_path'][0]
    except:
        group = None

    try:
        hdf5path = args['updateClass'][0]
    except:
        hdf5path = None
    
    if args['input'] != None:
        paths = [path.name for path in args['input']]
        print('Input found:')
        [print('\t'+path) for path in paths]

        for path in paths:

            print('Processing file:', path)

            if group is not None:
                savepath = args['group_path'][0]
                save_dir = os.path.dirname(savepath)
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                    assert os.path.exists(save_dir)
                group = True
            else:
                savepath = path.replace('.hdf5', '_metrics.tsv')
                savepath = savepath.replace('_reduced', '')
                group = False
            
            base = os.path.basename(path) #used in naming the indices to identify components

            if path.endswith('.hdf5'):
                assert path.endswith('_ica.hdf5') | path.endswith('_ica_reduced.hdf5'),\
                     "Path did not end in '_ica.hdf5'"

                print('\nLoading data to create classifier metrics\n------------------------------------------------')
                f = h5(path)

                if ('noise_components' not in f.keys()):
                    print('Calculating Noise Components')
                    noise, cutoff = sort_noise(f.load('timecourses'))
                    f.save({'noise_components':noise, 'cutoff':cutoff})
                else:
                    #Load data from ica.hdf5 file 
                    noise = f.load('noise_components')

                if 'artifact_components' in f.keys():
                    artifact = f.load('artifact_components')
                    if np.sum(artifact) != 0:
                        comb = noise + artifact
                        artifact[comb == 2] = 0 #if it is ID'd as noise and artifact, keep as noise 
                        neural = np.array(comb == 0).astype(int)
                    else:
                        neural = np.zeros_like(noise)
                else:
                    artifact = np.zeros_like(noise)
                    neural = np.zeros_like(noise)

                notnoise_index = np.where(noise==0)[0]

                if args['updateClass'] and np.sum(neural)!=0 and os.path.exists(savepath):
                    data = pd.read_csv(savepath, sep = '\t', index_col='exp_ic')
                    print('\nImporting dataframe\n------------------------------------')
                    print('Sum of each component BEFORE update:\n',  data[['artifact','neural']].sum())
                    data['artifact'] = artifact[notnoise_index]
                    data['neural'] = neural[notnoise_index]
                    print('\nSum of each component AFTER update:\n', data[['artifact','neural']].sum())
                    print('Saving dataframe to:', savepath)
                    data.to_csv(savepath, sep = '\t')
                    update = True

                if (os.path.exists(savepath) and args['force']) or (not os.path.exists(savepath)) or group:
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
                    data['neural'] = neural[notnoise_index]
                    try:
                        data['age'] = np.ones(int(neural[notnoise_index].shape[0])) * int(re.findall(r'\d+',meta['meta']['anml']['age'])[0])
                    except Exception as e:
                        print('ERROR: ', e)

                    data['exp_ic'] = [base[:-9] + '-' + '{}'.format(str(i).zfill(4)) for i in notnoise_index]
                    data = data.set_index('exp_ic')

                    eigenbrains = rebuild_eigenbrain(eig_vec, roimask=roimask, bulk=True)
                    del eig_vec

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
                            main_data = pd.concat([main_data, data])
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
                    savepath = path
                    print('Importing dataframe\n------------------------------------')
                    data = pd.read_csv(path, sep = '\t', index_col='exp_ic')
                    neural = data['neural']
                    # try:
                    print('Importing class distinctions from ICA file\n------------------------------------')
                    if os.path.exists(hdf5path):
                        f = h5(hdf5path)
                        
                        if ('noise_components' not in f.keys()) | args['force']:
                            print('Calculating Noise Components')
                            noise, cutoff = sort_noise(f.load('timecourses'))
                            f.save({'noise_components':noise, 'cutoff':cutoff})
                        else:
                            #Load data from ica.hdf5 file 
                            noise = f.load('noise_components')

                        if 'artifact_components' in f.keys():
                            artifact = f.load('artifact_components')
                            if np.sum(artifact) != 0:
                                comb = noise + artifact
                                artifact[comb == 2] = 0 #if it is id'd as noise and artifact, keep as noise 
                                neural_ica = np.array(comb == 0).astype(int)
                            else:
                                neural_ica = np.zeros_like(noise)
                        else:
                            artifact = np.zeros_like(noise)
                            neural_ica = np.zeros_like(noise)

                        notnoise_index = np.where(noise==0)[0]

                        base = os.path.basename(hdf5path) #used in naming the indices to identify components
                        indices = [base[:-9] + '-' + '{}'.format(str(i).zfill(4)) for i in notnoise_index]

                        print('Updating classifications')
                        data.loc[indices, 'artifact'] = artifact[notnoise_index]
                        data.loc[indices, 'neural'] = np.array(data['artifact'] == 0).astype(int)
                        update = True
                    else:
                        print("Could not find .hdf5 file.  Unable to update neural/artifact components.")

                except Exception as e:
                    print('Error importing dataFrame')
                    print('\t ERROR : ', e)
                    raise AssertionError('Could not import dataframe from csv file.')
            
            else:
                base = os.path.basename(path)
                print('\nFile type {} not understood.'.format(base))
                print('skipping ' + path + '\n')


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


 