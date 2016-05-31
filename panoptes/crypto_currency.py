from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import sys
import json
import datetime
import pandas as pd
import math

try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen

from builtins import open
from builtins import str
from builtins import int
from future import standard_library
standard_library.install_aliases()

def makequery(question):

    if question["type"] != "crypto_currency":
        print("Doesn't look like a crypto_currency query to me, can't do.")
        answer = {"useful": False}
        return answer
    else:
        min = sys.float_info.max
        mindate = ""
        maxdate = ""
        max = 0
        days = 0

        if "base" in question:
            base = question["base"]
        else:
            base = "USD"

        if question["currency"] == "BTC":
            source = "api.bitcoinaverage.com"
            # TODO add max informations
            url = "https://api.bitcoinaverage.com/history/" + base + "/per_day_all_time_history.csv"
            response = urlopen(url)
            reader = pd.read_csv(response)

            date1 = pd.to_datetime(question["datestart"]).strftime('%Y.%m.%d')
            date2 = pd.to_datetime(question["dateend"]).strftime('%Y.%m.%d')

            flag_after = False
            for index, row in reader.iterrows():
                date_from_csv = pd.to_datetime(row.datetime).strftime('%Y.%m.%d')
                if date_from_csv == date1:
                    flag_after = True

                if flag_after:
                    days += 1
                    if not math.isnan(row.low):
                        if min > row.low:
                            min = row.low
                            mindate = row.datetime

                    if not math.isnan(row.high):
                        if max < row.high:
                            max = row.high
                            maxdate = row.datetime

                if date_from_csv == date2:
                    break
            answer = {
                "minvalue": min,
                "minimum_on_date": mindate,
                "maximum_on_date": maxdate,
                "maxvalue": max,
                "days": days
            }
            print (answer)

        else:
            source = "coinmarketcap.northpole.ro"
            url = "http://coinmarketcap.northpole.ro/api/v5/history/" + question["currency"] + "_2016.json"
            response = urlopen(url)
            response = json.loads(response.read().decode("windows-1252"));

            date_to_call = pd.to_datetime(question["datestart"]).strftime('%d-%m-%Y')

            while question["datestart"] <= question["dateend"]:
                days += 1
                value = float(response["history"][date_to_call]["price"][base.lower()])

                if value < min:

                    min = value
                    mindate = date_to_call
                if value > max:

                    max = value
                    maxdate = date_to_call

                question["datestart"] = str(datetime.date(*(int(s) for s in question["datestart"].split('-'))) + datetime.timedelta(days=1))
                date_to_call = pd.to_datetime(question["datestart"]).strftime('%d-%m-%Y')

            answer = {
                "minvalue": min,
                "minimum_on_date": mindate,
                "maximum_on_date": maxdate,
                "maxvalue": max,
                "days": days
            }
    answer["useful"] = True
    answer["source"] = source
    return answer

if __name__ == "__main__":
    with open(sys.argv[1]) as que:
        question = json.load(que)
        print(makequery(question))
