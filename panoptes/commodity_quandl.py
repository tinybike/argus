from __future__ import print_function
import quandl
quandl.ApiConfig.api_key = "Vezn4-YcDxD5ihmyNuY-"
quandl.ApiConfig.api_version = '2015-04-09'
import sys
import json
import numpy as np
import pandas as pd
import datetime
from datetime import datetime


def help():
    print("\nFinds low and high value of a commodity, params are name of stock market\n"
          "(e.g. CME lumber for more columns or for one column(value) wgc gold) or company"
          "as source,\n second parameter is name of interested commodity, last two arguments "
          "are time period from-to \n\n"
          "usage: name_of_source name_of_commodity YYYY-MM-DD YYYY-MM-DD\n\n"
          "day value has to be in range <01-31>, month  <01-12>!\n\n"
          "For possible sources and their commodities see the list -l ")
    exit()

def list_sources(df):
    print ("\n")
    df1 = df.sort_values(['Source'], ascending=[True])
    for index, row in df1.iterrows():
        print ("source : "+row.Source+" , commodities : "+row.Name)
    print ("\n")
    exit()

def validate(date_text):
    try:
        return datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
        if date_text == 'marketstart' or date_text == 'marketend':
            return ''
        print("Wrong date format, please see the help below")
        return "Error"
        #help() TODO json

def future_date(date):
    if datetime.now() < date:
        print ("Future date was inserted : " + str(date))
        #exit("Future date was inserted : " + str(date)) # TODO


def check(code):
    if code is None:
        print ("For your source and commodity was not found any code in the list")
        #exit("For your source and commodity was not found any code in the list")

def isAfter(dateBefore, dateAfter):
    if dateBefore > dateAfter:
        print ("date_from is after date_to!")
        #exit("date_from is after date_to!")

def checkLiteralsDate(dateBefore, dateAfter):
    if dateBefore == 'marketend':
        exit("From date can not be marketend")  # TODO return instead of exit json

    if dateAfter == 'marketstart':
        exit("End date can not be marketstart")



def commodity_query(que):
    df = pd.read_csv('commodities.csv')
    if que["type"] != "commodity":
        print("Doesn't look like a commodity query to me, can't do.")
        answer = {"useful": False}
        return answer

    ''' if len(que) != 7:
        print ("Wrong number of arguments! For more help use argument -h")
        answer = {"error": True}'''

    source = None
    if "exchange" in question:
        source = str(que["exchange"])

    commodity = str(que["commodity"])
    date_from = que["datestart"]
    date_to = que["dateend"]

    # checkLiteralsDate(date_from, date_to) TODO json error

    date1 = validate(date_from)
    date2 = validate(date_to)

    if date1 != '' and date2 != '':
        isAfter(date1, date2)
        future_date(date2)

    # TODO what about something more clever then substring? Maybe just memory

    flag = False
    index_end = -1
    last_index = 0
    data = None
    while True:
        if source is not None:
            flag = True
        code, commodity_in_sentence, source, sector, index_end = search(source, commodity, index_end, df)

        # print ("CODE GET out of function :-:")
        # print (code)
        data = searchQuandl(date1, date2, code)

        if data is not None and not data.empty:  # if data was found
            break
        if flag:  # if the given source was wanted
            break

        if index_end == last_index:  # if search finished and nothing was found
            break

        last_index = index_end
        commodity_in_sentence = None
        code = None
        sector = None
        source = None

    # print (data)

    if len(data.columns) == 1:
        try:
            minimum = np.nanmin(data.Value.get_values())
            maximum = np.nanmax(data.Value.get_values())
        except TypeError:
            print("There is no value in the column for the given range")
        index_max = np.nanargmax(data.Value.get_values())
        date_max = data.index.get_values()[index_max]
        index_min = np.nanargmin(data.Value.get_values())
        date_min = data.index.get_values()[index_min]

        date_max = pd.to_datetime(str(date_max)).strftime('%Y.%m.%d')
        date_min = pd.to_datetime(str(date_min)).strftime('%Y.%m.%d')

        dump = {
            "useful": True,
            "minimum_on_date": date_min,
            "minvalue": minimum,
            "maximum_on_date": date_max,
            "maxvalue": maximum,
            "commodity name": commodity_in_sentence,
            "sector ": sector,
            "Quandl code ": code,
            "exchange ": source,
            "source": "Quandl data platform API"
        }
    else:
        try:
            minimum = np.nanmin(data.Settle.get_values())
            maximum = np.nanmax(data.Settle.get_values())
        except TypeError:
            print("There is no value in the column for the given range")
            # exit()

        index_max = np.nanargmax(data.Settle.get_values())
        # print("index")
        # print(index_max)
        date_max = data.index.get_values()[index_max]
        index_min = np.nanargmin(data.Settle.get_values())
        date_min = data.index.get_values()[index_min]
        print ("For more dimensions was chose Settle column")

        open_price_max = data.Open.get_values()[index_max]
        open_price_min = data.Open.get_values()[index_min]

        high_max = data.High.get_values()[index_max]
        high_min = data.High.get_values()[index_min]

        low_max = data.Low.get_values()[index_max]
        low_min = data.Low.get_values()[index_min]

        volume_max = data.Volume.get_values()[index_max]
        volume_min = data.Volume.get_values()[index_min]

        interest_max = data['Open Interest'].get_values()[index_max]
        interest_min = data['Open Interest'].get_values()[index_min]


        date_max = pd.to_datetime(str(date_max)).strftime('%Y.%m.%d')
        date_min = pd.to_datetime(str(date_min)).strftime('%Y.%m.%d')

        dump = {
            "useful": True,
            "minimum_on_date": date_min,
            "minvalue": minimum,  # settle
            "open_price_min": open_price_min,
            "high_price_min": high_min,
            "low_price_min": low_min,
            "volume_min": volume_min,
            "open_interest_min": interest_min,
            "maximum_on_date": date_max,
            "maxvalue": maximum,  # settle
            "open_price_max": open_price_max,
            "high_price_max": high_max,
            "low_price_max": low_max,
            "volume_max": volume_max,
            "open_interest_max": interest_max,
            "commodity name": commodity_in_sentence,
            "sector ": sector,
            "Quandl code ": code,
            "exchange ": source,
            "source": "Quandl data platform API",
        }

    #data.to_csv("test_data", sep='\t')
    #print (dump)
    return dump

def search(source, commodity, index_end, df):
    code = None
    commodity_in_sentence = None
    sector = None

    commodity = dictionary(commodity)

    if source is not None:
        for index, row in df.iterrows():
            if (row.Code.lower().find(commodity.lower()) >= 0) \
                    and row.Source.lower() == source.lower():
                code = row.Code
                commodity_in_sentence = row.Name
                sector = row.Sector

        if code is None:  # so we dont get code
            for index, row in df.iterrows():
                if (row.Name.lower().find(commodity.lower()) == 0
                    or row.Name.lower().find(" " + commodity.lower()) >= 0)\
                        and row.Source.lower() == source.lower():
                    code = row.Code
                    commodity_in_sentence = row.Name
                    sector = row.Sector
    else:

        for index, row in df.iterrows():
            if (row.Code.lower().find(commodity.lower()) >= 0)\
                    and index > index_end:
                source = row.Source
                code = row.Code
                commodity_in_sentence = row.Name
                sector = row.Sector
                index_end = index
                break
        # print(code)
        if code is None:  # so we dont get code
            for index, row in df.iterrows():
                if (row.Name.lower().find(commodity.lower()) == 0
                    or row.Name.lower().find(" " + commodity.lower()) >= 0)\
                        and index > index_end:
                    source = row.Source
                    code = row.Code
                    commodity_in_sentence = row.Name
                    sector = row.Sector
                    index_end = index
                    break

    # print("code with name : " + code)
    check(code)
    return code, commodity_in_sentence, source, sector, index_end

def searchQuandl(date1, date2, code):
    print ("date 1 ")
    print (date1)
    print ("date 2 ")
    print (date2)
    print (code)
    try:
        if date1 != '' and date2 != '':
            data = quandl.get(code, start_date=date1, end_date=date2)
            # OFDP/FUTURE_B1 OFDP has been deprecated all with OFDP
            # DOE/RWTC only to 2015, all wth DOE - MOVED TO EIA
            # WSJ/SOYB_OIL wrong code
            # JODI/OIL_LPSCKT_NIC is in Qunadl and is by our conditons, but its not in csv
            # data = quandl.get("DOE/RWTC", authtoken="Vezn4-YcDxD5ihmyNuY-", start_date="2015-03-19", end_date="2016-03-20")
        elif (date1 == '' and date2 != ''):
            data = quandl.get(code, end_date=date2)
        elif date1 != '' and date2 == '':
            data = quandl.get(code, start_date=date1)
        else:
            data = quandl.get(code)
    except Exception as e:
        print("The source/commodity was not found or the company/source does not exist anymore \n\n")
        print(e)
        data = None

    print ("Data --- ")
    print (data)
    print ("----------")
    #exit(0)
    return data

def dictionary(name):
    if name == "naturalgas":
        return "Natural Gas"
    return name

if __name__ == "__main__":
    arguments = sys.argv

    if len(sys.argv) == 3:
        name = arguments[2]
        if name == "-h":
            help()

    df = pd.read_csv('commodities.csv')
    if len(sys.argv) == 3:
        if name == "-l":
            list_sources(df)

    with open(sys.argv[1]) as que:
        question = json.load(que)
        print(commodity_query(question))
