import urllib.request
import sys
import json


def help():
    print("\nFinds low and high value of a yahoo stock, params are name of stock, time period from-to \n\n"
          "usage: finanget NAME YYYY-MM-DD YYYY-MM-DD\n\n"
          "day value of 0, month value of 0 (January = 1) or dates from the future cause weird behaviour in the API!\n")
    sys.exit(1)

def makequery(que):

    if que["type"] != "stock":
        return json.dumps({"Error": "Not a stock kinda question"})

    # Input parsing
    try:
        name = que["stock"]

        fdate = que["datestart"]
        tdate = que["dateend"]

        fy = str(fdate.split("-")[0])
        fm = str(fdate.split("-")[1])
        fd = str(fdate.split("-")[2])

        ty = str(tdate.split("-")[0])
        tm = str(tdate.split("-")[1])
        td = str(tdate.split("-")[2])
    except:
        help()

    #Query to the Yahoo finance API
    try:
        response = urllib.request.urlopen('http://ichart.finance.yahoo.com/table.csv?s='+name+'&a='+str(int(fm) - 1)+'&b='+fd+'&c='+fy+'&d='+str(int(tm) - 1)+'&e='+td+'&f='+ty+'&e=.csv')
    except:
        return json.dumps({"Error": "API request went badly"})

    records = []
    days = 0
    totalval = 0

    # Definition of the day's records class
    class dayrec:
        date = ""
        hi = float("0")
        lo = float("99999999999999")
        line = response.readline()


    # Reading through ALL of the response from yahoo api
    while True:
        line = response.readline()
        #print(line)
        if len(line) < 1:
            break

        #Parsing parts of the response
        days += 1
        d = dayrec()
        line = str(line).split("'")[1]
        d.date=(str(line).split(',')[0])
        d.hi=float(str(line).split(',')[2])
        d.lo=float(str(line).split(',')[3])
        records.insert(0,d)
        totalval += float(str(line).split(',')[6].split('\\')[0])

        # Initialise the extreme day records
        mini = dayrec
        maxi = dayrec

        # Check if the current one beats either of them
        for rec in records:
            if rec.hi > maxi.hi:
                maxi = rec
            if rec.lo < mini.lo:
                mini = rec


    # If we get a response with no business days (Most likely Saturday or Sunday), we have to recursively look back in time till we get at least one business day.
    if days == 0:
        try:
            recurdepth = que["recursion"]
        except:
            recurdepth = 0
        recurdepth += 1

        if recurdepth > 10:
            return json.dumps({"Error":"Recursion limit reached"})
        import datetime
        frdate = datetime.date(int(fy), int(fm), int(fd)) - datetime.timedelta(days=1)
        que[frdate] = frdate
        return makequery(que)


    # Parse the output as json
    dump = {
    "average_adj_closing":totalval/days,
    "trdays_in_period":days,
    "minimum_on_date":mini.date,
    "minvalue": mini.lo,
    "maximum_on_date":maxi.date,
    "maxvalue":maxi.hi
    }

    return json.dumps(dump)


if __name__ == "__main__":
    json.dumps()
    print(makequery(sys.argv))


