import urllib.request
import sys
import json


def help():
    print("\nFinds low and high value of a yahoo stock, params are name of stock, time period from-to \n\n"
          "usage: finanget NAME YYYY-MM-DD YYYY-MM-DD\n\n"
          "day value of 0, month value of 0 (January = 1) or dates from the future cause weird behaviour in the API!\n")
    sys.exit(1)

def makequery(question):
    if question["type"] != "stock":
        print("Doesn't look like a stock query to me, can't do.")
        answer = {"useful": False}
        return answer

    arguments = ['blurt', question["stock"], question["datestart"], question["dateend"]]

    try:
        name = arguments[1]

        fy = str(arguments[2]).split("-")[0]
        fm = str(arguments[2]).split("-")[1]
        fd = str(arguments[2]).split("-")[2]

        ty = str(arguments[3]).split("-")[0]
        tm = str(arguments[3]).split("-")[1]
        td = str(arguments[3]).split("-")[2]
    except:
        help()

    #Query to the Yahoo finance API
    response = urllib.request.urlopen('http://ichart.finance.yahoo.com/table.csv?s='+name+'&a='+str(int(fm) - 1)+'&b='+fd+'&c='+fy+'&d='+str(int(tm) - 1)+'&e='+td+'&f='+ty+'&e=.csv')

    records = []
    days = 0
    totalval = 0

    class dayrec:
        date = ""
        hi = float("0")
        lo = float("99999999999999")
        line = response.readline()

    while True:

        line = response.readline()
        #print(line)
        if len(line) < 1:
            break

        days += 1
        d = dayrec()
        line = str(line).split("'")[1]
        d.date=(str(line).split(',')[0])
        d.hi=float(str(line).split(',')[2])
        d.lo=float(str(line).split(',')[3])
        records.insert(0,d)
        totalval += float(str(line).split(',')[6].split('\\')[0])


    #If we get a response with no business days (Most likely Saturday or Sunday), we have to look back in time till we get at least one business day.
    if days == 0:
        try:
            recurdepth = question["recursive_search_for_real_bizday"]
        except:
            recurdepth = 0
        recurdepth += 1

        if recurdepth > 10:
            answer = {"useful": False,"error":"Recursion limit reached"}
            return answer
        import datetime
        #print("recursing: " + str(recurdepth))
        question["datestart"] = str(datetime.date(*(int(s) for s in question["datestart"].split('-'))) - datetime.timedelta(days=1))
        question["recursive_search_for_real_bizday"] = recurdepth
        return makequery(question)

    mini = dayrec
    maxi = dayrec



    for rec in records:
        if rec.hi > maxi.hi:
            maxi = rec
        if rec.lo < mini.lo:
            mini = rec

    dump = {
    "useful": True,
    "average_adj_closing":totalval/days,
    "trdays_in_period":days,
    "minimum_on_date":mini.date,
    "minvalue": mini.lo,
    "maximum_on_date":maxi.date,
    "maxvalue":maxi.hi,
    "source": "Yahoo time graph API"
    }

    #print(json.dumps(dump))
    return dump

if __name__ == "__main__":
    with open(sys.argv[1]) as que:
        question = json.load(que)
        print(makequery(question))




