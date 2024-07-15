from hmmlearn import hmm
from hmm_utils import *
from Data.dataImport import *
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser("Overcooked 2 argument parser")
    parser.add_argument("-t", "--ticker", type=str, required=False, help="Symbol for data retrieval")
    parser.add_argument("-r", "--resampleFreq", type=str, required=False, help="The interval of data, all lower cases (weekly, monthly...)")
    parser.add_argument("-f", "--format", type=str, required=False, help="Either csv or json")
    parser.add_argument("-s", "--start_date", type=str, required=True, help="Start date of training model")
    parser.add_argument("-e", "--end_date", type=str, required=True, help="End date for training model")
    parser.add_argument("-f", "--day_future", type=int, help="How many days to predict")
    parser.add_argument("-ts", "--test_size", type=float, help="The percentage of data for testing")
    parser.add_argument("-n", "--hidden_states", type=int, help="Number of hidden states for training")
    parser.add_argument("-nifc", "--n_interval_frac_change", type=int, help="Number of points for fractional change")
    parser.add_argument("-nifh", "--n_interval_frac_high", type=int, help="Number of points for high fractional change")
    parser.add_argument("nifl", "--n_interval_fractional_low", type=int, help="Number of points for low fractional change")
    parser.add_argument("-l", "--latency", type=int, help="Observation sequence duration")

    return parser.parse_args()

def main_loop(arglist):
    data_retrival(arglist.ticker, arglist.resampleFreq, arglist.format)

if __name__ == '__main__':
    arglist = parse_arguments()
    main_loop(arglist)


