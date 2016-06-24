try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen
import sys
import json
import datetime
import requests
from geopy.geocoders import Nominatim


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

        min = 99999999999999999999
        mindate = ""
        maxdate = ""
        max = 0
        days = 0

        if question["datestart"] == question["dateend"]:
            value = dayget(question)
            answer = {
                "minvalue": value["minimal"],
                "maxvalue": value["maximal"],
                "minimum_on_date": question["datestart"],
                "maximum_on_date": question["datestart"],
                "days": 1
            }
        else:
            while question["datestart"] <= question["dateend"]:
                days+=1
                value = dayget(question)
                if value["minimal"] < min:
                    min = value["minimal"]
                    mindate = question["datestart"]
                if value["maximal"] > max:
                    max = value["maximal"]
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
        answer["source"] = "forecast.io"
        return answer

def dayget(question):
    #print(question["datestart"])
    #print(("https://api.forecast.io/forecast/07b0a1ff17788bad43b9d3ad43819037/"+str(question["latitude"])+","+str(question["longitude"])+","+question["datestart"]+"T12:00:00"))
    forecast = requests.get("https://api.forecast.io/forecast/07b0a1ff17788bad43b9d3ad43819037/"+str(question["latitude"])+","+str(question["longitude"])+","+question["datestart"]+"T12:00:00?units=si")
    temperatures = {}
    temperatures["minimal"]=(forecast.json()["daily"]["data"][0]["temperatureMin"])
    temperatures["maximal"] = (forecast.json()["daily"]["data"][0]["temperatureMax"])
    return temperatures

if __name__ == "__main__":
    with open(sys.argv[1]) as que:
        question = json.load(que)
        print(makequery(question))
