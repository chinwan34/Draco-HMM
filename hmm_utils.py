from hmmlearn import hmm
import numpy as np
from sklearn.model_selection import train_test_split
import itertools
import pandas as pd

class HMMUtils:
    def __init__(self, ticker, resample_freq, format, day_future, test_size=0.33, n_hidden_states=4
                 ,n_intervals_frac_change=50, n_intervals_frac_high=10, n_intervals_frac_low=10,n_latency_days=10):
        data = pd.read_csv("/Users/roywan/Desktop/Draco/HMM-GMM/Data/{}_{}_{}.csv".format(ticker, resample_freq, format), delimiter=',')
        self.split_train_test_data(data, test_size)
        self.hmm = hmm.GaussianHMM(n_components=n_hidden_states)

        self.compute_all_possible_outcome(n_intervals_frac_change, n_intervals_frac_high, n_intervals_frac_low)
        self.days_in_future = day_future
        self.latency = n_latency_days


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
    
    def split_train_test_data(self, data, test_size):
        self.train_data, self.test_data = train_test_split(
            data, test_size=test_size, shuffle=False
        )
        
        self.train_data = self.train_data.drop(["divCash", "splitFactor", "volume", "adjClose", "adjHigh", "adjOpen", "adjVolume"])
        self.test_data = self.test_data.drop(["divCash", "splitFactor", "volume", "adjClose", "adjHigh", "adjOpen", "adjVolume"])
    
        self.days = len(self.test_data)

    def fit(self):
        observations = HMMUtils.extract_features(self.train_data)
        self.hmm.fit(observations)

    def compute_all_possible_outcome(self, 
                                     n_intervals_frac_change, 
                                     n_intervals_frac_high,
                                     n_intervals_frac_low):
        frac_change_range = np.linspace(-0.1, 0.1, n_intervals_frac_change)
        frac_high_range = np.linspace(0, 0.1, n_intervals_frac_high)
        frac_low_range = np.linspace(0, 0.1, n_intervals_frac_low)

        self.possible_outcomes = np.array(
            list(itertools.product(frac_change_range, frac_high_range, frac_low_range))
        )
    
    def get_most_probable_outcome(self, day_index):
        previous_start = max(0, day_index - self.latency)
        previous_end = max(0, day_index-1)
        previous_data = self.test_data.iloc[previous_start:previous_end]

        previous_data_features = HMMUtils.extract_features(previous_data)

        outcome_results = []

        for possible_outcome in self.possible_outcomes:
            total_data = np.row_stack((previous_data_features, possible_outcome))
            outcome_results.append(self.hmm.score(total_data))
        
        most_probable_outcome = self.possible_outcomes[np.argmax(outcome_results)]

        return most_probable_outcome
    