
import quandl
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
        print "Wrong date format, please see the help below"
        help()

def future_date(date):
    if datetime.now() < date:
        # print ("Future date was inserted : " + str(date))
        exit("Future date was inserted : " + str(date))

def check(code):
    if code is None:
        # print ("For your source and commodity was not found any code in the list")
        exit("For your source and commodity was not found any code in the list")

def isAfter(dateBefore, dateAfter):
    if dateBefore > dateAfter:
        # print ("date_from is after date_to!")
        exit("date_from is after date_to!")

def checkLiteralsDate(dateBefore, dateAfter):
    if dateBefore == 'marketend':
        exit("From date can not be marketend")

    if dateAfter == 'marketstart':
        exit("End date can not be marketstart")



def commodity_query(que, df):

    stringJson = ""
    for line in que:  # sys.stdin
        stringJson += line

    try:
        question = json.loads(stringJson)
    except ValueError as e:
        exit("Wrong json format!")  # maybe ''

    if question["type"] != "commodity":
        exit("Doesn't look like a commodity query to me, can't do.")

    # TODO make note on the wiki on this when it will be updated into module

    if len(question) != 7:
        print ("Wrong number of arguments! For more help use argument -h")
        help()

    source = str(question["exchange"])
    commodity = str(question["commodity"])
    date_from = question["datestart"]
    date_to = question["dateend"]

    checkLiteralsDate(date_from, date_to)

    date1 = validate(date_from)
    date2 = validate(date_to)

    if date1 != '' and date2 != '':
        isAfter(date1, date2)
        future_date(date2)

    code = None

    # TODO what about something more clever then substring? Maybe just memory
    for index, row in df.iterrows():
        if row.Code.lower().find(commodity.lower()) >= 0 and row.Source.lower() == source.lower():
            code = row.Code

    if code is None:  # so we dont get code
        for index, row in df.iterrows():
            if row.Name.lower().find(commodity.lower()) >= 0 and row.Source.lower() == source.lower():
                code = row.Code
        print ("code : "+code)
    # check(code)

    try:
        # auth_tok = "yourauthhere" as last argument for authentication if I will have account
        if date1 != '' and date2 != '':
            data = quandl.get(code, start_date=date1, end_date=date2)
        elif date1 == '' and date2 != '':
            data = quandl.get(code, end_date=date2)
        elif date1 != '' and date2 == '':
            data = quandl.get(code, start_date=date1)
        else:
            data = quandl.get(code)
    except Exception as e:
        print ("The source/commodity was not found or the company/source does not exist anymore \n\n")
        print (e)
        exit()

    print (data)
    if len(data.columns) == 1:
        try:
            minimum = np.nanmin(data.Value.get_values())
            maximum = np.nanmax(data.Value.get_values())
        except TypeError as e:
            print "There is no value in the column for the given range"
            exit()
        print data.Value.get_values()
        index_max = np.nanargmax(data.Value.get_values())
        # index_max = data.Value.get_values().index(maximum)
        date_max = data.index.get_values()[index_max]
        index_min = np.nanargmin(data.Value.get_values())
        date_min = data.index.get_values()[index_min]
    else:
        try:
            minimum = np.nanmin(data.Settle.get_values())
            maximum = np.nanmax(data.Settle.get_values())
        except TypeError as e:
            print "There is no value in the column for the given range"
            exit()
        # index_max = data.Settle.get_values().index(maximum)
        print data.columns
        print type(data.Settle.get_values())
        print type(data)

        index_max = np.nanargmax(data.Settle.get_values())
        # index_max = data.Value.get_values().index(maximum)
        print "index"
        print index_max
        date_max = data.index.get_values()[index_max]
        index_min = np.nanargmin(data.Settle.get_values())
        date_min = data.index.get_values()[index_min]
        print ("For more dimensions was chose Settle column")

    answer = {"Questioned value": str(question["value"])}
    answer['Source'] = "Quandl data platform API"

    print date_max
    date_max = pd.to_datetime(str(date_max)).strftime('%Y.%m.%d')
    date_min = pd.to_datetime(str(date_min)).strftime('%Y.%m.%d')
    if question["comp"] == "above":
        if maximum > question["value"]:
            answer["Decision"] = True
            answer["Maximal value"] = str(maximum)
            answer["On Date"] = date_max
        else:
            answer["Decision"] = False
            answer["Maximal value"] = str(maximum)
            answer["On Date"] = date_max

    if question["comp"] == "below":
        if minimum < question["value"]:
            answer["Decision"] = True
            answer["Minimal value"] = str(minimum)
            answer["On Date"] = date_min
        else:
            answer["Decision"] = False
            answer["Maximal value"] = str(minimum)
            answer["On Date"] = date_min

    return json.dumps(answer)

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
        print(commodity_query(que, df))
