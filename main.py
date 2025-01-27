from hmmlearn import hmm
from hmm_utils import *
from Data.dataImport import *
import argparse
from os.path import exists

def parse_arguments():
    parser = argparse.ArgumentParser("Argument parser")
    parser.add_argument("-t", "--ticker", type=str, required=True, help="Symbol for data retrieval")
    parser.add_argument("-r", "--resampleFreq", type=str, required=True, help="The interval of data, all lower cases (weekly, monthly...)")
    parser.add_argument("-f", "--format", type=str, required=True, help="Either csv or json")
    parser.add_argument("-s", "--start_date", type=str, required=True, help="Start date of training model")
    parser.add_argument("-e", "--end_date", type=str, required=True, help="End date for training model")
    parser.add_argument("-df", "--day_future", type=int, required=False, default=None, help="How many days to predict")
    parser.add_argument("-ts", "--test_size", type=float, help="The percentage of data for testing")
    parser.add_argument("-n", "--hidden_states", type=int, help="Number of hidden states for training")
    parser.add_argument("-ran", "--random_state", type=int, default=None, help="The random seed for the Gaussian HMM")
    parser.add_argument("-da", "--days_average", type=int, help="The average number of days in interval")

    # Not yet checked
    parser.add_argument("-nifc", "--n_interval_frac_change", type=int, help="Number of points for fractional change")
    parser.add_argument("-nifh", "--n_interval_frac_high", type=int, help="Number of points for high fractional change")
    parser.add_argument("-nifl", "--n_interval_fractional_low", type=int, help="Number of points for low fractional change")


    parser.add_argument("-l", "--latency", type=int, help="Observation sequence duration")
    parser.add_argument("-m", "--metrics", action="store_true", default=False, help="Boolean for metric display")
    parser.add_argument("-p", "--plot", action="store_true", default=False, help="Boolean for plotting results")
    parser.add_argument("-pm", "--predict_mode", action="store_true", default=False, help="If predict mode is utilized, test_data percentage returns to")

    return parser.parse_args()

def main_loop(arglist):
    # if not exists("/Users/roywan/Desktop/Draco/HMM-GMM/Data/{}_{}_{}".format(arglist.ticker, arglist.resampleFreq, arglist.start_date)):  
    data_retrival(arglist.ticker, arglist.resampleFreq, arglist.format, arglist.start_date)

    predictor = HMMUtils(arglist=arglist)
    predictor.fit()
    print("Fit finished")
    # predictor.predict_close_price_average(arglist.days_average)

    # return

    if arglist.metrics:
        # temp_list = predictor.real_close_prices()["close"].tolist()
        # predicted_close = temp_list[:-arglist.day_future]
        # predicted_close.extend(predictor.predict_close_prices_future_days())
        actual_close = predictor.real_close_prices()
        predicted_close, predicted_win_lose = predictor.predict_close_price_for_period()
        actual_close = predictor.real_close_prices()
        if actual_close.iloc[-1].isnull().any():
            actual_close = actual_close.iloc[:-1]

        actual_close["Predicted_Close"] = predicted_close
        actual_close["Predicted_Win_Lose"] = predicted_win_lose

        output_df = actual_close.rename(columns={"close": "Actual_Close"})
        output_df.to_csv("/Users/roywan/Desktop/Draco/HMM-GMM/Data/Predicted_result/{}_{}_{}_{}".format(arglist.ticker, arglist.resampleFreq, actual_close.iloc[0]["date"], arglist.random_state))

        mse = predictor.calc_mse(output_df)
        accuracy = predictor.calc_accuracy()
        print("MSE Result: ", mse)
        print("Accuracy Result: ", accuracy)

    if not exists ("/Users/roywan/Desktop/Draco/HMM-GMM/Data/Predicted_result/MSE_test"):
        mse_df = pd.DataFrame(columns=["Seed", "Sample_Date", "MSE", "Accuracy"])
    else:
        mse_df = pd.read_csv("/Users/roywan/Desktop/Draco/HMM-GMM/Data/Predicted_result/MSE_test", delimiter=',')
    if not ((mse_df['Seed'] == arglist.random_state) & (mse_df["Sample_Date"] == output_df.iloc[0]["date"]) & (mse_df["MSE"] == mse) & (mse_df["Accuracy"] == accuracy)).any():
        new_index = [[arglist.random_state, output_df.iloc[0]["date"], mse, accuracy]]
        df_extended = pd.DataFrame(new_index, columns=['Seed', 'Sample_Date', 'MSE', 'Accuracy'])
        mse_df = pd.concat([mse_df, df_extended], ignore_index=True)
            
        mse_df.to_csv("/Users/roywan/Desktop/Draco/HMM-GMM/Data/Predicted_result/MSE_test", index=False)

    
        if arglist.plot:
            predictor.plot_results(output_df, arglist.ticker, arglist.day_future)


    if arglist.day_future:        
        predictor.populate_future_days()
        future_pred_close = predictor.predict_close_prices_future_days()
        predictor.final_prediction_strategy(future_pred_close)



if __name__ == '__main__':
    arglist = parse_arguments()
    main_loop(arglist)


