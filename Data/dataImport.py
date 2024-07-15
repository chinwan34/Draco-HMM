import requests

api_key = "49859e3a95b4d7278db3909b95cb4ae0f81a63be"

def data_retrival(ticker, resampleFreq, format):
    if ticker == "" or resampleFreq == "" or format == "":
        print("Not enough input for data retrieval")
        return 

    ticker_used = ticker
    resample_freq_used = resampleFreq
    format_used = format

    requestResponse = requests.get("https://api.tiingo.com/tiingo/daily/{}/prices?startDate=2019-07-09&format={}&resampleFreq={}&token={}".format(ticker_used, format_used, resample_freq_used, api_key))

    if requestResponse.status_code == 200:
        with open('/Users/roywan/Desktop/Draco/HMM-GMM/Data/{}_{}_{}'.format(ticker_used, resample_freq_used, format_used), 'wb') as fileSave:
            fileSave.write(requestResponse.content)
    else:
        print("Error occurred, please check parameters.")

