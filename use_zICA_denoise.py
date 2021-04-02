import struct, sys, os, pickle, random
from tqdm import tqdm
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt

from zICA_denoise import zICA_denoise

def fread_bynary(rb,ndx,bsize,blen):
    rb.seek(ndx*bsize,0)
    a = rb.read(bsize*blen)
    return np.frombuffer(a, dtype='i2').astype(float)
    
def plot_channels(X,offset=100):
    n_ch = X.shape[1]
    for i in range(n_ch):
        plt.plot(X[:,i] + offset * i,linewidth=0.5);
        
def clip_verylarge(X,maxz):
    # detect average z-amplitude > maxz
    X_shp = X.shape
    X = X.reshape((X_shp[0]*X_shp[1],))
    Xz = zscore(X)
    std_X = np.std(X)
    X[np.where(Xz > maxz)] = maxz * std_X
    X[np.where(Xz < -maxz)] = -maxz * std_X
    return X.reshape((X_shp[0],X_shp[1]))

# set parameters
in_filename = sys.argv[1]
out_filename = sys.argv[2]
n_channels = int(sys.argv[3])

th_skw_ic = float(sys.argv[4]) if len(sys.argv) > 4 else 1.
n_ic = int(sys.argv[5]) if len(sys.argv) > 5 else n_channels
b_save_results = int(sys.argv[6]) if len(sys.argv) > 6 else 0
max_std_clip  = int(sys.argv[7]) if len(sys.argv) > 7 else 50
n_chunk = int(sys.argv[8]) if len(sys.argv) > 8 else 1000000
n_sample_fit = int(sys.argv[9]) if len(sys.argv) > 9 else 300
dur_sample_fit = int(sys.argv[10]) if len(sys.argv) > 10 else 10000
rand_seed = int(sys.argv[11]) if len(sys.argv) > 11 else None

save_dir = 'results_zica_denoise'

print('[start zICA denoising]')

print('input file: ', in_filename)
print('output file: ', out_filename)
print('n_channels: ', str(n_channels))
print('th_skw_ic: ', str(th_skw_ic))
print('n_ic: ', str(n_ic))
print('b_save_results: ', str(b_save_results))
print('max_std_clip: ', str(max_std_clip))
print('n_chunk: ', str(n_chunk))
print('n_sample_fit: ', str(n_sample_fit))
print('dur_sample_fit: ', str(dur_sample_fit))
print('rand_seed: ', str(rand_seed))

if rand_seed is not None:
    random.seed(rand_seed)

# open trace file (binary file, ex. .dat file)
NBbytes = 2
bytesperndx = int(n_channels*NBbytes)
rb = open(in_filename, 'rb')
rb.seek(0,2)
n_sample_all = int(rb.tell()/bytesperndx)

# read smple data for fit ica
print('read file for fitting model...')
rand_ndx = random.sample(range(n_sample_all-dur_sample_fit-1), n_sample_fit)
X = np.zeros((n_sample_fit*dur_sample_fit, n_channels))
for i in tqdm(range(len(rand_ndx))):
    i_X = i * dur_sample_fit
    ndx_start = rand_ndx[i]
    X[i_X:i_X+dur_sample_fit,:] = fread_bynary(rb,ndx_start,bytesperndx,dur_sample_fit).reshape(dur_sample_fit, n_channels)
X = clip_verylarge(X,max_std_clip)

# calc ica
print('calcurating ica...')
Idn = zICA_denoise(n_channels=n_channels, n_ic=n_ic, rand_seed=rand_seed)
ics = Idn.fit_ica_models(X)
skw_ics, std_ics = Idn.calc_skw_ics(ics)
del ics

# check results
if b_save_results:
    print('plotting example traces...')
    
    # make dir to save
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # read sample trace data
    ndx_start = 2*n_chunk # pick up 3rd chunk
    X = fread_bynary(rb,ndx_start,bytesperndx,n_chunk).reshape(n_chunk, n_channels)
    X = clip_verylarge(X,max_std_clip)
    
    # std distribution
    n_show = 3000;
    n_grid_show = int(np.ceil(np.sqrt(std_ics.shape[1]+1)));
    plt.figure(figsize=(30,15))
    for i in range(std_ics.shape[1]):
        plt.subplot(n_grid_show, n_grid_show, i+1)
        plt.hist(std_ics[:,i], bins=30)
        plt.title('ch std dist: IC' + str(i))
    plt.subplot(n_grid_show, n_grid_show, i+2)
    plt.hist(skw_ics, bins=30, color = 'red')
    plt.title('Skewness of all IC')
    
    # pc explained variance ratio
    plt.subplot(n_grid_show, n_grid_show, i+3)
    plt.plot(np.cumsum(Idn.pca_part(X).explained_variance_ratio_))
    plt.hlines([0.7, 0.8, 0.9], 0, n_ic, 'black', linestyles='dashed')
    plt.ylim(0,1)
    plt.title('PC explained variance ratio')
    plt.savefig(save_dir+'/dist_std.png')
    plt.close()
    
    # trace
    clm_ics = np.array(range(n_ic))
    for i in range(n_ic):
        plt.figure(figsize=(30,15))
        plot_channels(Idn.denoise(X,cut_ndx_ics = np.where(clm_ics != i))[0:n_show,:],offset=500)
        plt.title('IC ' + str(i) + ', skw: ' + '{:.2f}'.format(skw_ics[i]))
        plt.savefig(save_dir+'/trace_ic'+str(i)+'.png')
        plt.close()
            
    plt.figure(figsize=(30,15))
    plt.subplot(1,2,1)
    plot_channels(X[0:n_show,:],offset=500)
    plt.title('Original')
    plt.subplot(1,2,2)
    plot_channels(Idn.denoise(X,cut_ndx_ics = np.where(skw_ics < th_skw_ic))[0:n_show,:],offset=500)
    plt.title('Denoised')
    plt.savefig(save_dir+'/trace_original_denoised.png')
    plt.close()

# apply denoise and save
print('denoising by generated model and saving the results...')
n_loop = int(np.ceil(n_sample_all / n_chunk))
wb = open(out_filename, 'wb')
for i in tqdm(range(n_loop)):
    ndx_start = i*n_chunk
    i_chunk = n_chunk
    if (i+1)*n_chunk > n_sample_all:
        i_chunk = n_sample_all - (i*n_chunk)
    X = fread_bynary(rb,ndx_start,bytesperndx,i_chunk).reshape(i_chunk, n_channels)
    X = clip_verylarge(X,max_std_clip)
    
    ### <may not need ######################
    #Idn = ICA_denoise(n_channels=n_channels,n_ic=n_ic)
    #ics, pcs = Idn.fit_ica_models(X)
    #skw_ics, std_ics = Idn.calc_skw_ics(ics)
    ### may not need> ######################
    
    X_denoised = Idn.denoise(X,cut_ndx_ics = np.where(skw_ics < th_skw_ic)).reshape((n_channels*i_chunk,)).astype('i2')

    bg = X_denoised.tobytes()
    wb.write(bg)

rb.close()
wb.close()

if b_save_results:
    # save model
    with open(save_dir+'/model.pkl', 'wb') as f:
     pickle.dump(Idn, f)

print('[finished zICA denoising]')
