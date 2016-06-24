import sys
import json
from stock import makequery as stockquery
from currency import makequery as currencyquery
from commodity_quandl import commodity_query as commodityquery
from crypto_currency import makequery as cryptoquery
from weather import makequery as weatherquery

def proccess(que):
    squery = stockquery(que)

    if squery['useful'] == True:
        return evaluate(que, squery)

    cquery = commodityquery(que)
    if cquery['useful'] == True:
        return evaluate(que, cquery)

    currquery = currencyquery(que)
    if currquery['useful'] == True:
        return evaluate(que, currquery)

    crypto_currquery = cryptoquery(que)
    if crypto_currquery['useful'] == True:
        return evaluate(que, crypto_currquery)

    wquery = weatherquery(que)
    if wquery['useful'] == True:
        return evaluate(que, wquery)

def evaluate(question, response):
    answer = {}
    answer["source"] = response["source"]

    print(response)

    if "comp" in question:
        if "compwith" not in question:
            if question["comp"] == "above":
                question["compwith"] = "maximum"
            else:
                question["compwith"] = "minimum"

        if question["compwith"] == "maximum":
            answer["Questioned value"] = question["value"]
            if question["comp"] == "above":
                if response["maxvalue"] > question["value"]:
                    answer["decision"] = True
                    answer["maximal value"] = str(response["maxvalue"])
                    answer["on Date"] = response["maximum_on_date"]
                else:
                    answer["decision"] = False
                    answer["maximal value"] = str(response["maxvalue"])
                    answer["on Date"] = response["maximum_on_date"]

            if question["comp"] == "below":
                if response["maxvalue"] < question["value"]:
                    answer["decision"] = True
                    answer["maximal value"] = str(response["maxvalue"])
                    answer["on Date"] = response["maximum_on_date"]
                else:
                    answer["decision"] = False
                    answer["maximal value"] = str(response["maxvalue"])
                answer["on Date"] = response["maximum_on_date"]

        if question["compwith"] == "minimum":
            answer["Questioned value"] = question["value"]
            if question["comp"] == "above":
                if response["minvalue"] > question["value"]:
                    answer["decision"] = True
                    answer["minimal value"] = str(response["minvalue"])
                    answer["on Date"] = response["maximum_on_date"]
                else:
                    answer["decision"] = False
                    answer["minimal value"] = str(response["minvalue"])
                    answer["on Date"] = response["maximum_on_date"]

            if question["comp"] == "below":
                if response["minvalue"] < question["value"]:
                    answer["decision"] = True
                    answer["minimal value"] = str(response["minvalue"])
                    answer["on Date"] = response["minimum_on_date"]
                else:
                    answer["decision"] = False
                    answer["minimal value"] = str(response["minvalue"])
                answer["on Date"] = response["minimum_on_date"]
    answer.update(response)

    return json.dumps(answer)

if __name__ == "__main__":
    with open(sys.argv[1]) as que:
        question = json.load(que)
        print(proccess(question))
