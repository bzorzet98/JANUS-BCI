import numpy as np
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, mutual_info_classif

class CSP_classifier:
    def __init__(self, n_components=4, classifier=None, reg=None, 
                 log=True, norm_trace=False, component_order='mutual_info'):
        """
        n_components: Number of spatial filters.
        classifier: Sklearn classifier (defaults to LDA).
        """
        # component_order='mutual_info' helps pick the most discriminative filters
        self.csp = CSP(n_components=n_components, reg=reg, log=log, 
                       norm_trace=norm_trace, component_order=component_order)
        self.clf = classifier if classifier is not None else LDA()
        
    def fit(self, X, y):
        """
        X: shape (n_epochs, n_channels, n_times)
        y: shape (n_epochs,)
        """
        # MNE CSP returns the log-variance features directly when log=True
        csp_features = self.csp.fit_transform(X, y)
        self.clf.fit(csp_features, y)
        return self

    def transform(self, X):
        return self.csp.transform(X)

    def predict(self, X):
        features = self.transform(X)
        return self.clf.predict(features)
    
    def predict_proba(self, X):
        """
        Returns the probability of each class.
        X: shape (n_epochs, n_channels, n_times)
        Returns: (n_epochs, n_classes)
        """
        features = self.transform(X)
        return self.clf.predict_proba(features)
    
    def score(self, X, y):
        features = self.transform(X)
        return self.clf.score(features, y)
    

class FBCSP_classifier:
    def __init__(self, sfreq=250, 
                 bands=None, 
                 n_components=4,
                 k_features=4, 
                 use_selector=True,
                 order = 4):
        """
        sfreq: Sampling frequency.
        bands: List of (low, high) tuples.
        n_components: CSP components per band.
        k_features: Number of features to select (if use_selector is True).
        use_selector: Whether to use Mutual Information feature selection.
        """
        self.sfreq = sfreq
        self.n_components = n_components
        self.k_features = k_features
        self.use_selector = use_selector
        self.bands = bands if bands is not None else [(f, f+4) for f in range(1, 40, 4)]
        self.order = order
        self.csp_bank = [CSP(n_components=n_components, reg=None, log=True) for _ in self.bands]
        
        if self.use_selector:
            self.selector = SelectKBest(mutual_info_classif, k=k_features)
            self.clf = LDA() # Standard LDA
        else:
            self.selector = None
            # If no selector, use Shrinkage LDA for stability with many features
            self.clf = LDA(solver='eigen', shrinkage='auto')

    def _filter_data(self, X, l_freq, h_freq):
        from scipy.signal import butter, filtfilt
        nyq = 0.5 * self.sfreq
        freqcut = (l_freq / nyq, h_freq / nyq) 
        b, a = butter(self.order, freqcut, btype="bandpass")
        X_ = filtfilt(b, a, X, axis=-1)
        return X_
        
    def fit(self, X, y):
        features_list = []
        for i, (l_freq, h_freq) in enumerate(self.bands):
            X_filt = self._filter_data(X, l_freq, h_freq)
            feat = self.csp_bank[i].fit_transform(X_filt, y)
            features_list.append(feat)
            
        X_features = np.hstack(features_list)
        
        if self.use_selector:
            X_features = self.selector.fit_transform(X_features, y)
            
        self.clf.fit(X_features, y)
        return self

    def transform(self, X):
        features_list = []
        for i, (l_freq, h_freq) in enumerate(self.bands):
            X_filt = self._filter_data(X, l_freq, h_freq)
            feat = self.csp_bank[i].transform(X_filt)
            features_list.append(feat)
        
        X_features = np.hstack(features_list)
        if self.use_selector:
            X_features = self.selector.transform(X_features)
        return X_features

    def predict(self, X):
        features = self.transform(X)
        return self.clf.predict(features)

    def predict_proba(self, X):
        """
        Returns the probability of each class.
        X: shape (n_epochs, n_channels, n_times)
        Returns: (n_epochs, n_classes)
        """
        features = self.transform(X)
        return self.clf.predict_proba(features)

    def score(self, X, y):
        return self.clf.score(self.transform(X), y)