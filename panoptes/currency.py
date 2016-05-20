import urllib.request
import sys
import json


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
        return dayget(question)

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

    return response

if __name__ == "__main__":
    with open(sys.argv[1]) as que:
        question = json.load(que)
        print(makequery(question))
