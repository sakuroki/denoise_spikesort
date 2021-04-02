import numpy as np
from scipy import linalg
from mountainlab_pytools import mdaio
from ml_ms4alg import ms4alg
from scipy import stats
from sklearn.covariance import MinCovDet

import random, os, sys, json

'''
def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.tile(np.mean(data,axis=0),(x.shape[0],1))
    if not cov:
        cov = np.cov(data.T)
    inv_covmat = linalg.inv(cov)
    mahal = (x_minus_mu.dot(inv_covmat) * x_minus_mu).sum(-1)
    return mahal
'''

#def separate_outlier():
# input: result of ms4alg (times, labels, channels), input (time trace), distance_threshold, clip_size

path_timeseries = sys.argv[1] #path_timeseries = 'fil3.mda'
path_firing_in = sys.argv[2] #path_firing_in = 'ms_test_result_03.mda'
path_firing_out = sys.argv[3] #path_firing_out = 'ms_test_result_03_nc.mda'


if os.path.exists(path_firing_out):
    print('[Output file is already exist.]')
    print('[Exit separate_outlier.py]')
    sys.exit()

distance_mad_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 8
#distance_p_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 8
clip_size = int(sys.argv[5]) if len(sys.argv) > 5 else 11
num_features = int(sys.argv[6]) if len(sys.argv) > 6 else 10
adjacency_radius = float(sys.argv[7]) if len(sys.argv) > 7 else -1
max_num_clips_sample = int(sys.argv[8]) if len(sys.argv) > 8 else 1000
remain_outlier = int(sys.argv[9]) if len(sys.argv) > 9 else 0 # boolean 0 or 1

path_geom_in = sys.argv[10] if len(sys.argv) > 10 else None

print('path_timeseries: ', path_timeseries)
print('path_firing_in: ', path_firing_in)
print('path_geom_in: ', path_geom_in)
print('ppath_firing_out: ', path_firing_out)
print('distance_mad_threshold: {} (mad is median version std)'.format(distance_mad_threshold))
print('clip_size: {}'.format(clip_size))
print('num_features: {}'.format(num_features))
print('adjacency_radius: {}'.format(adjacency_radius))
print('max_num_clips_sample: {}'.format(max_num_clips_sample))
print('remain_outlier: {}'.format(remain_outlier))

print('[Start separate_outlier]')

# read spike sorting results
Y=mdaio.readmda(path_firing_in)
channels = Y[0,:].astype('int32')
times = Y[1,:].astype('int64')
labels = Y[2,:].astype('int32')
unique_labels = np.unique(labels)

# read timeseries file
if path_timeseries[-4:] == '.prv':
    fr = open(path_timeseries,mode='r')
    jtxt = fr.read()
    fr.close()
    path_ts = json.loads(jtxt)['original_path']
elif path_timeseries[-4:] == '.mda':
    path_ts = path_timeseries
    
temp_hdf5_path='timeseries.hdf5'
if os.path.exists(temp_hdf5_path):
    os.remove(temp_hdf5_path)
hdf5_chunk_size=1000000
hdf5_padding=clip_size*10
ms4alg.prepare_timeseries_hdf5(path_ts, temp_hdf5_path,chunk_size=hdf5_chunk_size,padding=hdf5_padding)
X=ms4alg.TimeseriesModel_Hdf5(temp_hdf5_path)

M_global=X.numChannels()
N=X.numTimepoints()
if path_geom_in is not None:
    geom = np.genfromtxt(path_geom_in, delimiter=',')
else:
    geom = np.zeros((M_global,2))
    
chunk_infos=ms4alg.create_chunk_infos(N=N,chunk_size=100000)
#chunk_infos=create_chunk_infos(N=N,chunk_size=10000000)
# loop per labels
label_outlier = np.max(unique_labels) + 1 if remain_outlier else 0

# remove small spike clusters
for il in unique_labels:
    times_il = times[np.where(labels == il)]
    if len(times_il) < 5:
        labels[np.where(labels == il)] = 0
        labels[np.where(labels > il)] = labels[np.where(labels > il)] - 1
        print("Removed small spike cluster: #{}, {} spikes.".format(il,len(times_il)))

unique_labels = np.unique(labels)
if unique_labels[0] == 0:
    unique_labels = unique_labels[1:]
#print(unique_labels)

for il in unique_labels:
    
    print('Calc outlier cell #{}.'.format(il))
    
    # pick up labeled times
    times_il = times[np.where(labels == il)]
    
    # channels
    channels_il = channels[np.where(labels == il)]
    m_central = channels_il[0] - 1
    nbhd_channels = ms4alg.get_channel_neighborhood(m_central,geom,adjacency_radius=adjacency_radius)
    #nbhd_channels = np.arange(m_central-2,m_central+3)
    #nbhd_channels = nbhd_channels[nbhd_channels>=0]
    #nbhd_channels = nbhd_channels[nbhd_channels<M_global]
    M_neigh=len(nbhd_channels)
    m_central_rel=np.where(nbhd_channels==m_central)[0][0]

    print('Neighboorhood of channel {} has {} channels.'.format(m_central,M_neigh))
    
    # comput features (pick up waveform and calc PCA)
    features = ms4alg.compute_event_features_from_timeseries_model(X,times_il,nbhd_channels=nbhd_channels,clip_size=clip_size,max_num_clips_for_pca=max_num_clips_sample,num_features=num_features*2,chunk_infos=chunk_infos)
    
    # calc mahalanobis distance form centroid of the cluster
    if len(times_il) > max_num_clips_sample:
        rand_idx = random.sample(range(features.shape[1]), max_num_clips_sample)
        features_for_conv = features[:,rand_idx]
    else:
        features_for_conv = features
    #maha_dist = mahalanobis(features.T, features_for_conv.T)
    
    # Minimum Covariance Determinant
    mcd = MinCovDet()
    mcd.fit(features_for_conv.T)
    maha_dist = mcd.mahalanobis(features.T)
    
    #import matplotlib.pyplot as plt
    #print(np.median(maha_dist))
    #plt.hist(maha_dist,range=(0,200))
    #plt.show()
    
    # pick up outlier and name new label
    distance_threshold =  stats.median_absolute_deviation(maha_dist) * distance_mad_threshold + np.median(maha_dist)
    #distance_threshold = stats.chi2.ppf(1-distance_p_threshold, features.shape[0])
    times_outliner = times_il[np.where(maha_dist >= distance_threshold)]
    
    if times_outliner.size > 0:
        for t in times_outliner:
            labels[np.where(np.logical_and(times == t, labels == il))] = label_outlier
    else:
        if remain_outlier:
            labels[np.where(np.logical_and(times == times_il[np.argmax(maha_dist)], labels == il))] = label_outlier # only one spike which has maxmum mahalanobis distance were separated to adjust order of cleaned and noise clusters
    label_outlier = label_outlier + 1 if remain_outlier else 0
    
    print('Median: {:.2f}, Threshold: {:.2f} (Mahalanobis Distance)'.format(np.median(maha_dist),distance_threshold))
    print('Num of Outliers is {} in {} spikes.'.format(times_outliner.size,times_il.size))
    
# save result
ms4alg.write_firings_file(channels,times,labels,path_firing_out)
os.remove(temp_hdf5_path)
print('[Done separate_outlier]')
