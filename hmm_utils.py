from hmmlearn import hmm
import numpy as np
from sklearn.model_selection import train_test_split

class HMMUtils:
    def __init__(ticker, resample_freq, format):
        data = np.genfromtxt("/Users/roywan/Desktop/Draco/HMM-GMM/Data/{}_{}_{}.csv".format(ticker, resample_freq, format), delimiter=',')
        
        pass

    def GMM_HMM(self, O, lengths, n_states, v_type, n_iter, verbose=True):
        model = hmm.GaussianHMM(n_components=n_states, covariance_type=v_type, n_iter=n_iter, verbose=verbose).fit(O, lengths)

        return model
    
    def extract_features(self, data):
        open_price = np.array(data["open"])
        close_price = np.array(data["close"])
        high_price = np.array(data["high"])
        low_price = np.array(data["low"])
        
        frac_change = (close_price - open_price) / open_price
        frac_high = (high_price - open_price) / open_price
        frac_low = (open_price - low_price) / open_price

        return np.column_stack((frac_change, frac_high, frac_low))
    
    def split_train_test_data(self, test_size):
        