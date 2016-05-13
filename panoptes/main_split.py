import sys
import json

from finanget import stockquery
nodice = json.dumps({"Error": "found no reasonable answer"})


def proccess(que):

    ans = stockquery(que)
    if "Decision" in ans:
        return ans
    else:
        return nodice

def evaluate(que):

    question = json.load(que)

    params = ['blurt',question["stock"], question["datestart"], question["dateend"]]
    finanget_response = json.loads(makequery(params))

    answer = {"Questioned value": str(question["value"])}
    answer['Source'] = "Yahoo time graph API"

    if "Error" in finanget_response:
        answer["Error"] = finanget_response["Error"]
        return answer

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
        print(proccess(que))
