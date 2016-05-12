import sys
import json
from finanget import makequery

def stockquery(que):

    question = json.load(que)
    if question["type"] != "stock":
        print("Doesn't look like a stock query to me, can't do.")
        sys.exit(1)

    #print(question["type"])
    #print(question["stock"])
    #print(question["stock"] + " " + question["datestart"] + " " + question["dateend"])

    params = ['blurt',question["stock"], question["datestart"], question["dateend"]]

    #print(params)
    #print(makequery(params))
    finanget_response = json.loads(makequery(params))
    #print(finanget_response)

    answer = {"Questioned value": str(question["value"])}
    answer['Source'] = "Yahoo time graph API"

    if question["comp"] == "above":
        if finanget_response["maxvalue"] > question["value"]:
            answer["Decision"] = True
            answer["Maximal value"] = str(finanget_response["maxvalue"])
            answer["On Date"] = finanget_response["maximum_on_date"]
        else:
            answer["Decision"] = False
            answer["Maximal value"] =  str(finanget_response["maxvalue"])
            answer["On Date"] =  finanget_response["maximum_on_date"]

    if question["comp"] == "below":
        if finanget_response["minvalue"] < question["value"]:
            answer["Decision"] = True
            answer["Minimal value"] = str(finanget_response["minvalue"])
            answer["On Date"] = finanget_response["minimum_on_date"]
        else:
            answer["Decision"] = False
            answer["Maximal value"] = str(finanget_response["minvalue"])
            answer["On Date"] = finanget_response["minimum_on_date"]

    return json.dumps(answer)

if __name__ == "__main__":
    with open(sys.argv[1]) as que:
        print(stockquery(que))


