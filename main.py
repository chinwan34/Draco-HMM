from hmmlearn import hmm
from hmm_utils import *
from Data.dataImport import *
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser("Overcooked 2 argument parser")
    parser.add_argument("--ticker", type=str, required=False, help="Symbol for data retrieval")
    parser.add_argument("--resampleFreq", type=str, required=False, help="The interval of data, all lower cases (weekly, monthly...)")
    parser.add_argument("--format", type=str, required=False, help="Either csv or json")

    return parser.parse_args()

def main_loop(arglist):
    data_retrival(arglist.ticker, arglist.resampleFreq, arglist.format)

if __name__ == '__main__':
    arglist = parse_arguments()
    main_loop(arglist)


