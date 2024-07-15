from hmmlearn import hmm
import numpy as np
from sklearn.model_selection import train_test_split
import itertools
import pandas as pd
from tqdm import tqdm
from datetime import timedelta

class HMMUtils:
    def __init__(self, ticker, resample_freq, format, day_future, start_date, end_date, test_size=0.33, n_hidden_states=4
                 ,n_intervals_frac_change=50, n_intervals_frac_high=10, n_intervals_frac_low=10,n_latency_days=10):
        data = pd.read_csv("/Users/roywan/Desktop/Draco/HMM-GMM/Data/{}_{}_{}.csv".format(ticker, resample_freq, format), delimiter=',')
        self.split_train_test_data(data, test_size)
        
        # Currently avoided initial training for initial probabilities
        self.hmm = hmm.GaussianHMM(n_components=n_hidden_states)

        self.compute_all_possible_outcome(n_intervals_frac_change, n_intervals_frac_high, n_intervals_frac_low)
        self.days_in_future = day_future
        self.latency = n_latency_days
        self.predicted_close = None


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
            # Compute through hmmlearn module
            outcome_results.append(self.hmm.score(total_data))
        
        most_probable_outcome = self.possible_outcomes[np.argmax(outcome_results)]

        return most_probable_outcome
    
    def predict_close_price(self, day_index):
        open_price = self.test_data.iloc[day_index]["open"]
        (
            predicted_frac_change,
            _,
            _,
        ) = self.get_most_probable_outcome(day_index)

        return open_price * (1+predicted_frac_change)
    
    def predict_close_price_for_period(self):
        predicted_close_prices = []

        for day_index in tqdm(range(self.days)):
            predicted_close_prices.append(self.predict_price(day_index))
        self.predicted_close = predicted_close_prices

        return predicted_close_prices

    def populate_future_days(self):
        last_day = self.test_data.index[0] + timedelta(days=self.days_in_future)
        future_dates = pd.date_range(
            self.test_data.index[0] + pd.offsets.DateOffset(1), last_day
        )
        new_df = pd.DataFrame(
            index=future_dates, columns=["high", "low", "open", "close"]
        )

        self.test_data = pd.concat([self.test_data, new_df])
        # Replace opening price of 1st day in future with close price of last day
        self.test_data.iloc[self.days]["open"] = self.test_data.iloc[self.days-1]["close"]

    def predict_close_price_future_days(self, day_index):
        # Only predict a particular day and write into self.test_data
        open_price = self.test_data.iloc[day_index]["open"]

        (
            predicted_frac_change,
            pred_frac_high,
            pred_frac_low,
        ) = self.get_most_probable_outcome(day_index)
        predicted_close_price = open_price * (1+predicted_frac_change)

        self.test_data.iloc[day_index]["close"] = predicted_close_price
        self.test_data.iloc[day_index]["high"] = open_price * (1 + pred_frac_high)
        self.test_data.iloc[day_index]["low"] = open_price * (1 - pred_frac_low)

        return predicted_close_price

    def predict_close_prices_future_days(self):
        predicted_close_prices = []
        future_indices = len(self.test_data) - self.days_in_future

        for day_index in tqdm(future_indices, len(self.test_data)):
            predicted_close_prices.append(self.predict_close_price_future_days(day_index))
            try:
                self.test_data.iloc[day_index+1]["open"] = self.test_data.iloc[day_index]["close"]
            except IndexError:
                continue
        
        self.predicted_close = predicted_close_prices
        return self.predicted_close
    
    def real_close_prices(self):
        return self.test_data.loc[:, ["close"]]
    
    def calc_mse(self):
        
    




