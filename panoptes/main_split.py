import sys
import json
from stock import makequery as stockquery

def proccess(que):
    squery = stockquery(que)
    if squery['useful'] == True:
        return evaluate(que, squery)

def evaluate(question, response):
    answer = {}
    answer["source"] = response["source"]

    if "comp" in question:
        answer["Questioned value"] = question["value"]
        if question["comp"] == "above":
            if response["maxvalue"] > question["value"]:
                answer["decision"] = True
                answer["maximal value"] = str(response["maxvalue"])
                answer["on Date"] = response["maximum_on_date"]
            else:
                answer["decision"] = False
                answer["maximal value"] =  str(response["maxvalue"])
                answer["on Date"] =  response["maximum_on_date"]

        if question["comp"] == "below":
            if response["minvalue"] < question["value"]:
                answer["decision"] = True
                answer["minimal value"] = str(response["minvalue"])
                answer["on Date"] = response["minimum_on_date"]
            else:
                answer["decision"] = False
                answer["maximal value"] = str(response["minvalue"])
                answer["on Date"] = response["minimum_on_date"]

    return json.dumps(answer)

if __name__ == "__main__":
    with open(sys.argv[1]) as que:
        question = json.load(que)
        print(proccess(question))
