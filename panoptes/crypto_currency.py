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

        if "base" in question:
            base = question["base"]
        else:
            base = "USD"

        if question["currency"] == "BTC":
            source = "api.bitcoinaverage.com"
            answer = BTCquerry(base, question)

        else:
            source = "coinmarketcap.northpole.ro"
            answer = coin_market_api(base, question)
    answer["useful"] = True
    answer["source"] = source
    return answer

def BTCquerry(base, question):

    min = sys.float_info.max
    mindate = ""
    maxdate = ""
    max = 0
    days = 0

    average_min = 0
    volume_min = 0
    average_max = 0
    volume_max = 0
    # and in github check its done
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
                    average_min = row.average
                    volume_min = row.volume

            if not math.isnan(row.high):
                if max < row.high:
                    max = row.high
                    maxdate = row.datetime
                    average_max = row.average
                    volume_max = row.volume

        if date_from_csv == date2:
            break
    answer = {
        "minvalue": min,
        "minimum_on_date": mindate,
        "average_at_date_min": average_min,
        "volume_at_date_min": volume_min,
        "maximum_on_date": maxdate,
        "average_at_date_max": average_max,
        "volume_at_date_max": volume_max,
        "maxvalue": max,
        "days": days
    }
    return answer

def coin_market_api(base, question):
    # HERE CAN BE ADDED A LOTS OF ANOTHER INFORMATIONS - availableSupply, availableSupplyNumber, volume24
    # change1h, change7h, change7d
    min = sys.float_info.max
    mindate = ""
    maxdate = ""
    max = 0
    days = 0
    url = "http://coinmarketcap.northpole.ro/api/v5/history/" + question["currency"] + "_2016.json"
    response = urlopen(url)
    response = json.loads(response.read().decode("windows-1252"));

    market_cap_max = 0
    market_cap_min = 0

    date_to_call = pd.to_datetime(question["datestart"]).strftime('%d-%m-%Y')
    while question["datestart"] <= question["dateend"]:
        days += 1
        value = float(response["history"][date_to_call]["price"][base.lower()])
        if value < min:
            market_cap_min = response["history"][date_to_call]["marketCap"][base.lower()]
            min = value
            mindate = date_to_call
        if value > max:
            market_cap_max = response["history"][date_to_call]["marketCap"][base.lower()]
            max = value
            maxdate = date_to_call

        question["datestart"] = str(
            datetime.date(*(int(s) for s in question["datestart"].split('-'))) + datetime.timedelta(days=1))
        date_to_call = pd.to_datetime(question["datestart"]).strftime('%d-%m-%Y')

    answer = {

        "minvalue": min,
        "minimum_on_date": mindate,
        "market_max_at_date": market_cap_min,
        "maximum_on_date": maxdate,
        "market_min_at_date": market_cap_max,
        "maxvalue": max,
        "days": days
    }
    return answer

if __name__ == "__main__":
    with open(sys.argv[1]) as que:
        question = json.load(que)
        print(makequery(question))
