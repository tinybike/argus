import sys
import json
from finanget import makequery

def proccess(que):
    stockquery = makequery(que)
    if stockquery['useful'] == True:
        return evaluate(que, stockquery)

def evaluate(question, response):
    answer = {}
    answer["Questioned value"] = question["value"]
    answer['Source'] = "Yahoo time graph API"

    if question["comp"] == "above":
        if response["maxvalue"] > question["value"]:
            answer["Decision"] = True
            answer["Maximal value"] = str(response["maxvalue"])
            answer["On Date"] = response["maximum_on_date"]
        else:
            answer["Decision"] = False
            answer["Maximal value"] =  str(response["maxvalue"])
            answer["On Date"] =  response["maximum_on_date"]

    if question["comp"] == "below":
        if response["minvalue"] < question["value"]:
            answer["Decision"] = True
            answer["Minimal value"] = str(response["minvalue"])
            answer["On Date"] = response["minimum_on_date"]
        else:
            answer["Decision"] = False
            answer["Maximal value"] = str(response["minvalue"])
            answer["On Date"] = response["minimum_on_date"]

    return json.dumps(answer)

if __name__ == "__main__":
    with open(sys.argv[1]) as que:
        question = json.load(que)
        print(proccess(question))
