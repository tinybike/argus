try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen
import sys
import json
import datetime
from geopy.geocoders import Nominatim
from forecastio import *


class dayrec(object):
    date = ""
    hi = float("0")
    lo = float("99999999999999")

def makequery(question):

    if question["type"] != "weather":
        print("Doesn't look like a weather query to me, can't do.")
        answer = {"useful": False}
        return answer
    else:
        geolocator = Nominatim()
        location = geolocator.geocode(question["place"])

        question["Address"] = location.address
        question["latitude"] = location.latitude
        question["longitude"] = location.longitude

        print("Decoded position as: " + question["Address"])
        print(question["latitude"])
        print(question["longitude"])

        min = 99999999999999999999
        mindate = ""
        maxdate = ""
        max = 0
        days = 0

        if question["datestart"] == question["dateend"]:
            value = dayget(question)
            answer = {
                "minvalue": value,
                "maxvalue": value,
                "minimum_on_date": question["datestart"],
                "maximum_on_date": question["datestart"],
                "days": 1
            }
        else:
            while question["datestart"] <= question["dateend"]:
                days+=1
                value = dayget(question)
                if value < min:
                    min = value
                    mindate = question["datestart"]
                if value > max:
                    max = value
                    maxdate = question["datestart"]
                question["datestart"] = str(datetime.date(*(int(s) for s in question["datestart"].split('-'))) + datetime.timedelta(days=1))
            answer = {
                "minvalue": min,
                "minimum_on_date": mindate,
                "maximum_on_date": maxdate,
                "maxvalue": max,
                "days": days
            }
        answer["useful"] = True
        answer["source"] = "fixer.io"
        return answer

def dayget(question):



if __name__ == "__main__":
    with open(sys.argv[1]) as que:
        question = json.load(que)
        print(makequery(question))
