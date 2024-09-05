from hmmlearn import hmm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import itertools
import pandas as pd
from tqdm import tqdm
from datetime import timedelta
import matplotlib.pyplot as plt
import datetime
from dateutil import parser
import xgboost as xgb

class HMMUtils:
    def __init__(self, arglist, test_size=0.098, n_hidden_states=4
                 ,n_intervals_frac_change=50, n_intervals_frac_high=10, n_intervals_frac_low=10,n_latency_days=10):
        data = pd.read_csv("/Users/roywan/Desktop/Draco/HMM-GMM/Data/{}_{}_{}".format(arglist.ticker, arglist.resampleFreq, arglist.start_date), delimiter=',')
        self.arglist = arglist
        self.split_train_test_data(data, test_size)
        
        # Currently avoided initial training for initial probabilities
        self.hmm = hmm.GaussianHMM(n_components=n_hidden_states, random_state=arglist.random_state)

        # If going into XGB
        if arglist.xgb:
            S, A, gamma = self.XGB_pre_process()
            A, self.hmm, pi = self.XGB_XMM(S,A,gamma)

        self.compute_all_possible_outcome(n_intervals_frac_change, n_intervals_frac_high, n_intervals_frac_low)
        self.days_in_future = arglist.day_future

        if arglist.latency:
            self.latency = arglist.latency
        else:
            self.latency = n_latency_days
        self.predicted_close = None
    
    def XGB_pre_process(self, O):
        pi = self.hmm.startprob_
        A = self.hmm.transmat_
        _, S = self.hmm.decode(O, algorithm='viterbi')  
        gamma = self.hmm.predict_proba(O)  # Predict posterior in every state    

        return S, A, gamma

    def XGB_HMM(self,O,S,A,gamma):
        n_states = 4
        stop_flag = 0
        iteration = 1
        log_likelihood = -np.inf
        min_delta = 1e-4
        model = 1

        prior_pi = np.array([sum(S == i) / len(S) for i in range(n_states)])
        B_matrix = gamma / prior_pi

        record_log_likelihood = []
        best_result = []

        while stop_flag <= 3:
            A, gamma = self.prob_re_estimate(A, B_matrix, prior_pi, lengths)
            B_matrix = gamma / prior_pi

            new_S, _, new_log_likelihood = self.xgb_prediction(B_matrix, lengths, A, prior_pi)
            record_log_likelihood.append(new_log_likelihood)

            if len(best_result) == 0:
                best_result = [A, model, prior_pi, new_log_likelihood]
            elif new_log_likelihood > best_result[3]:
                best_result = [A, model, prior_pi, new_log_likelihood]
                temp = gamma
            
            # Check for re-evaluate ending
            if new_log_likelihood - log_likelihood <= min_delta:
                stop_flag += 1
            else:
                stop_flag = 0
            
            log_likelihood = new_log_likelihood
            iteration += 1

        model = self.xgb_model(O, temp, n_states)
        best_result[1] = model

        return best_result[0], best_result[1], best_result[2]
        
    def xgb_model(self, X, gamma, n_states):
        # The model parameters from 
        params = {'objective': 'multi:softprob',
              'learning_rate': 0.01,
              'colsample_bytree': 0.886,
              'min_child_weight': 3,
              'max_depth': 10,
              'subsample': 0.886,
              'reg_alpha': 1.5,  
              'reg_lambda': 0.5,  
              'gamma': 0.5, 
              'n_jobs': -1,
              'eval_metric': 'mlogloss',
              'scale_pos_weight': 1,
              'random_state': 201806,
              'missing': None,
              'silent': 1,
              'max_delta_step': 0,
              'num_class': n_states}

        y = np.array([np.argmax(i) for i in gamma])
        temp = np.array([np.max(i) for i in gamma])
        y = y[temp >= 0.9]
        X = X[temp >= 0.9]

        sample_weight = temp[temp >= 0.9]
        # Just a matrix specified for XGBs
        d_train = xgb.DMatrix(X, y, weight=sample_weight)

        model = xgb.train(params, d_train, num_boost_round=1000)
        pred = np.array([np.argmax(i) for i in model.predict(d_train)])

        print(sum(pred==y)/len(y))

        return model

    def form_index(self, lengths, index):
        begin = sum(lengths[0:index])
        end = begin + lengths[index]

        return begin, end

    def prob_re_estimate(self, A, B, pi, lengths):
        """
        A: (n_states, n_states): Transition Probability Matrix
        B: (n_samples, n_states): Emission Probability Matrix

        """
        n_states = B.shape[1]
        n_samples = B.shape[0]

        # Probability of being in state S_i at time k and observe sequence up to k
        alpha_all = np.zeros((n_samples, n_states))
        # Probability of observing sequence from k+1 to T, given currently in S_i at time k
        beta_all = np.zeros((n_samples, n_states))
        # Intermediate calculation
        di_gamma_all = np.zeros((n_samples, n_states, n_states))
        gamma_all = np.zeros((n_samples, n_states))
        scale_all = np.zeros(n_samples)

        for k in range(len(lengths)):
            begin, end = self.form_index(lengths, k)
            T = end - begin
            B_copy = B[begin:end].copy()
            alpha = np.zeros((T, n_states))
            beta = np.zeros((T, n_states))
            di_gamma = np.zeros((T, n_states, n_states))
            gamma = np.zeros((T, n_states))
            scale = np.zeros(T)

        # Step 2.1: Alpha calculation (Forward Algorithm)
        for i in range(n_states):   # Initialization
            alpha[0, i] = pi[i] * B[0, i]
        scale[0] = sum(alpha[0])

        for t in range(1, T):
            for i in range(n_states):
                alpha[t, i] = 0
                for j in range(n_states):
                    # THe inside equation
                    alpha[t, i] += alpha[t-1, j] * A[j, i]
                alpha[t, i] = alpha[t, i] * B[t, i]
            scale[t] = 1/sum(alpha[t])
            alpha[t] = alpha[t] * scale[t]

        # Step 2.2: Beta calculation (Backward Algorithm)
        beta[T-1] = scale[T-1]
        for t in range (T-2, -1, -1):
            for i in range(n_states):
                beta[t,i] = 0
                for j in range(n_states):
                    beta[t,i] += A[i,j] * B[t+1, j] * beta[t+1, j]
                beta[t, i] = scale[t] * beta[t, i]
        
        # Step 2.3: Calculate Gamma
        for t in range(T-1):
            for i in range(n_states):
                gamma[t, i] = 0
                for j in range(n_states):
                    di_gamma[t, i, j] = alpha[t, i] * A[i, j] * B[t+1, j] * beta[t+1, j]
                    gamma[t, i] += di_gamma[t, i, j]
        
        t = T-1
        gamma[t] = alpha[t]
        alpha_all[begin:end] = alpha
        beta_all[begin:end] = beta
        di_gamma_all[begin:end] = di_gamma
        gamma_all[begin:end] = gamma
        scale_all[begin:end] = scale

        # Step 3.1 Re-estimate matrix A
        # Summing All gamma in di_gamma over all gamma in gamma
        for i in range(n_states):
            for j in range(n_states):
                num = 0
                den = 0
                for k in range(len(lengths)):
                    begin, end = self.form_index(lengths, k)
                    num += np.sum(di_gamma_all[begin:end, i, j])
                    den += np.sum(gamma_all[begin:end, i])
                A[i, j] = num / den
        
        return A, gamma_all

    def xgb_prediction(self, B, lengths, A, pi):
        # Viterbi Algorithm prediction for comparisons
        # Log-ikelihood means how well the current model explains the sequence of emissions

        log_likelihood_list = []
        n_states = len(pi)
        init_flag = 1

        for i in range(len(lengths)):
            begin, end = self.form_index(lengths, i)
            B_copy = B[begin:end].copy()
            curr_state = np.zeros(lengths[i])
            curr_state_prob = np.zeros((lengths[i], n_states))

            for j in range(lengths[i]):
                if j == 0:
                    curr_state_prob[j] = B_copy[j] * pi
                else:
                    for k in range(n_states):
                        temp = curr_state_prob[j-1] * A[:, k] * B_copy[j, k]
                        curr_state_prob[j, k] = max(temp)
                curr_state_prob[j] = curr_state_prob[j] / np.sum(curr_state_prob[j])
                # Update the maximum value at current step
                curr_state[j] = np.argmax(curr_state_prob[j])
            
            for j in range(lengths[i]):
                if j == 0:
                    curr_log_likelihood = np.log(pi[int(curr_state[j])]) + np.log(B_copy[j, int(curr_state[j])])
                else:
                    curr_log_likelihood += np.log(A[int(curr_state[j-1]), int(curr_state[j])]) + np.log(B_copy[j, int(curr_state[j])])
            
            if init_flag == 1:
                state = curr_state
                state_prob = curr_state_prob
                init_flag = 0
            else:
                state = np.hstack((state, curr_state))
                state_prob = np.row_stack((state_prob, curr_state_prob))
            
            log_likelihood_list.append(curr_log_likelihood)
        
        log_likelihood = 0
        for i in log_likelihood_list:
            log_likelihood += i
        
        return state, state_prob, log_likelihood
        

    def GMM_HMM(self, O, lengths, n_states, v_type, n_iter, verbose=True):
        model = hmm.GaussianHMM(n_components=n_states, covariance_type=v_type, n_iter=n_iter, verbose=verbose).fit(O, lengths)

        return model
    
    @staticmethod
    def extract_features(data):
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
        
        self.train_data = self.train_data.drop(["volume", "adjClose", "adjHigh", "adjLow", "adjOpen", "adjVolume", "divCash", "splitFactor"], axis=1)
        self.test_data = self.test_data.drop(["volume", "adjClose", "adjHigh", "adjLow", "adjOpen", "adjVolume", "divCash", "splitFactor"], axis=1)
    
        self.days = len(self.test_data)

    def fit(self):
        observations = HMMUtils.extract_features(self.train_data)
        self.hmm.fit(observations)

        # Need to save to pickle file

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
        # print("Most probable Outcome", most_probable_outcome)

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
        predicted_win_lose = []

        for day_index in tqdm(range(self.days)):
            close_price = self.predict_close_price(day_index)

            predicted_close_prices.append(close_price)
            if self.test_data.iloc[day_index]["open"] < close_price:
                predicted_win_lose.append(1)
            else:
                predicted_win_lose.append(-1)
            print(self.test_data.iloc[day_index]["date"])
        self.predicted_close = predicted_close_prices

        return predicted_close_prices, predicted_win_lose

    def predict_close_price_average(self, days):
        # Currently not in Main, low accuracy, require improvements
        predicted_close_prices = []
        predicted_win_lose = []
        correct_wrong = []
        counter = 0

        for day_index in tqdm(self.range_help(self.days, days)):
            close_price = self.predict_close_price(day_index)
            predicted_close_prices.append(close_price)
            open_avg = (self.test_data.iloc[day_index:day_index+days]["open"].sum())/days
            close_avg = (self.test_data.iloc[day_index:day_index+days]["close"].sum())/days

            if open_avg < close_price:
                predicted_win_lose.append(1)
            else:
                predicted_win_lose.append(-1)
            
            if (open_avg > close_avg and open_avg > close_price) or (open_avg < close_avg and open_avg < close_price):
                correct_wrong.append(1)
                counter+=1
            else:
                correct_wrong.append(0)                

            print(self.test_data.iloc[day_index]["date"])
            print("Current %: ", counter/len(correct_wrong))
        self.predicted_close = predicted_close_prices
        accuracy = sum([1 for x in correct_wrong if x == 1]) / len(correct_wrong)
        print("Accuracy for {} day average: ".format(days), accuracy)

        return predicted_close_prices, predicted_win_lose

    def range_help(self, end, step):
        i = 0
        while i < end:
            yield i
            i += step 
        yield end - 1


    def populate_future_days(self):
        last_day = self.test_data.index[-1] + self.days_in_future
        future_dates = pd.Index(range(self.test_data.index[-1], last_day))
        # last_day = datetime.datetime.strptime(self.test_data.iloc[-1]["date"], "%Y-%m-%d").date() + timedelta(days=self.days_in_future)
        # future_dates = pd.date_range(
        #     datetime.datetime.strptime(self.test_data.iloc[-1]["date"], "%Y-%m-%d").date() + pd.offsets.DateOffset(1), last_day
        # )
        new_df = pd.DataFrame(
            index=future_dates, columns=["high", "low", "open", "close"]
        )
        self.test_data = pd.concat([self.test_data, new_df])

        # Replace opening price of 1st day in future with close price of last day
        self.test_data.reset_index(drop=True, inplace=True)

        self.test_data.loc[self.days, "open"] = self.test_data.loc[self.days-1, "close"]
        # self.test_data.fillna(0, inplace=True)

    def add_one_day(self, date):
        date = datetime.datetime.strptime(date, "%Y-%m-%d")
        new_date = date + timedelta(days=1)
        return new_date.strftime("%Y-%m-%d")

    def predict_close_price_future_days(self, day_index):
        # Only predict a particular day and write into self.test_data
        open_price = self.test_data.iloc[day_index]["open"]

        (
            predicted_frac_change,
            pred_frac_high,
            pred_frac_low,
        ) = self.get_most_probable_outcome(day_index)
        predicted_close_price = open_price * (1+predicted_frac_change)

        print("CURRENT INDEX", day_index)
        new_date = self.add_one_day(self.test_data.loc[day_index-1]["date"])
        self.test_data.loc[day_index, "date"] = new_date
        self.test_data.loc[day_index, "close"] = predicted_close_price
        self.test_data.loc[day_index, "high"] = open_price * (1 + pred_frac_high)
        self.test_data.loc[day_index, "low"] = open_price * (1 - pred_frac_low)

        return predicted_close_price

    def predict_close_prices_future_days(self):
        predicted_close_prices = []
        future_indices = len(self.test_data) - self.days_in_future

        self.checkpoint = self.test_data.loc[future_indices-1, "close"]

        for day_index in tqdm(range(future_indices, len(self.test_data))):
            predicted_close_prices.append(self.predict_close_price_future_days(day_index))
            try:
                self.test_data.loc[day_index+1, "open"] = self.test_data.iloc[day_index]["close"]
            except IndexError:
                continue
        
        self.predicted_close = predicted_close_prices
        return self.predicted_close
    
    def final_prediction_strategy(self, future_pred_close):
        if future_pred_close[-1] > self.checkpoint:
            print("Buy signal detected: {}".format(1))
        elif future_pred_close[-1] == self.checkpoint:
            print("Hold signal detected: {}".format(0))
        else:
            print("Sell signal detected: {}".format(-1))

    def real_close_prices(self):
        return self.test_data.loc[:, ["date", "close"]]
    
    def calc_mse(self, df):
        actual_array = (df.loc[:, "Actual_Close"]).values
        pred_array = (df.loc[:, "Predicted_Close"]).values
        mse = mean_squared_error(actual_array, pred_array)
        return mse
    
    def plot_results(self, in_df, stock_name, day_future):
        in_df = in_df.reset_index()  # Required for plotting
        # in_df = in_df.tail(day_future)
        ax = plt.gca()
        in_df.plot(kind="line", x="date", y="Actual_Close", ax=ax)
        in_df.plot(kind="line", x="date", y="Predicted_Close", color="red", ax=ax)
        plt.ylabel("Daily Close Price (in USD)")
        plt.title(str(stock_name) + " daily closing stock prices")
        # save_dir = f"{out_dir}/{stock_name}_results_plot.png"
        # plt.savefig(save_dir)
        plt.show()
        plt.close("all")
    
    def prediction_correctness(self, x):
        if (x["Actual_Close"] > x["open"] and x["Predicted_Close"] > x["open"]) or (x["Actual_Close"] < x["open"] and x["Predicted_Close"] < x["open"]):
            return 1
        else:
            return 0

    def calc_accuracy(self):
        df = pd.read_csv("/Users/roywan/Desktop/Draco/HMM-GMM/Data/Predicted_result/{}_{}_{}_{}".format(self.arglist.ticker, self.arglist.resampleFreq, self.test_data.iloc[0]["date"], self.arglist.random_state))
        df = df.assign(open=self.test_data.loc[:, "open"].values)
        new_df = df.apply(self.prediction_correctness, axis=1).reset_index(name='correct_prediction')
        df = pd.concat([df, new_df], axis=1)
        df.to_csv("/Users/roywan/Desktop/Draco/HMM-GMM/Data/Predicted_result/New_{}_{}_{}_{}".format(self.arglist.ticker, self.arglist.resampleFreq, self.test_data.iloc[0]["date"], self.arglist.random_state), index=False)
        accuracy = (df["correct_prediction"] == 1).sum() / (len(df.index))
        return accuracy