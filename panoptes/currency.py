import urllib.request
import sys
import json
import datetime


class dayrec(object):
    date = ""
    hi = float("0")
    lo = float("99999999999999")

def makequery(question):

    if question["type"] != "currency":
        print("Doesn't look like a currency query to me, can't do.")
        answer = {"useful": False}
        return answer
    else:
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

    if "base" in question:
        base = "?base=" +question["base"]
    else:
        base = "?base=USD"
    symb = ";symbols=" + question["currency"]
    print("http://api.fixer.io/"+question["datestart"]+base+symb)
    res = urllib.request.urlopen("http://api.fixer.io/"+question["datestart"]+base+symb)
    response = json.loads(res.read().decode("windows-1252"));
    toss, value = response["rates"].popitem()
    return value

if __name__ == "__main__":
    with open(sys.argv[1]) as que:
        question = json.load(que)
        print(makequery(question))
