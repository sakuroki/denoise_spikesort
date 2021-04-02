import numpy as np
from scipy.stats import skew
from sklearn.decomposition import PCA, FastICA

class ICA_denoise:
    def __init__(self, n_channels, n_ic=None, rand_seed=None):
        self.n_channels = n_channels
        if n_ic is None:
            self.n_ic = n_channels
        else:
            self.n_ic = n_ic
        self.rand_seed = rand_seed
            
        self.ica = FastICA(n_components = self.n_ic, random_state=self.rand_seed)
    
    def get_ica_model(self):
        return self.ica
        
    def calc_skw_ics(self, ics):
        self.std_ics = np.zeros((self.n_channels, self.n_ic))
        self.skw_ics = np.zeros((self.n_ic,))
        clm_ics = np.array(range(self.n_ic))
        for i in range(self.n_ic):
            ics_use = ics.copy()
            ics_use[:,np.where(clm_ics!=i)] = 0 # choose ic to use
            X_ici = self.ica.inverse_transform(ics_use)
            self.std_ics[:,i] = np.std(X_ici, axis=0) # recunstruct signals
            self.skw_ics[i] = skew(self.std_ics[:,i])
        return self.skw_ics, self.std_ics
    
    def get_skw_ics(self):
        return self.skw_ics
    def get_std_dist_ics(self):
        return self.std_ics

    def fit_ica_models(self, X):
        ics = self.ica.fit_transform(X)
        return ics
        
    def transform_ica(self, X):
        ics = self.ica.transform(X)
        return ics
    
    def denoise(self,X,cut_ndx_ics):
        ics = self.ica.transform(X)
        ics[:,cut_ndx_ics] = 0;
        return self.ica.inverse_transform(ics)
        
    def pca_part(self,X):
        pca = PCA(n_components = self.n_ic, random_state=self.rand_seed)
        pca.fit_transform(X)
        return pca
