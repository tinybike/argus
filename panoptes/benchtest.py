import csv
import json
import sys
import main_split

def parsecsv(csvfile):
    questions = csv.reader(csvfile)

    total = 0
    APIhit = 0
    decided = 0

    bulk = []

    for row in questions:
        queststring = ""
        for rowy in row:
            if "{" in rowy:
                print(queststring)
                print(rowy)
                total+=1
                question = json.loads(rowy)
                question["questionstring"] = queststring
                result,answer = testcall(question)
                pack = question,result,answer
                bulk+=pack
                print(result)
                print("|----------------------------------------------|\n")


                if result["APIres"]:
                    APIhit+=1
                if result["decided"]:
                    decided+=1
            else:
                queststring = rowy

    print("Results:\nTotal questions: " + str(total) + "\nAPI hits: " + str(APIhit) + "\nDecisions: " + str(decided) + "\nDecision success rate: " + str(100*decided/total)+"%\n")
    print(bulk)

def testcall(question):
    try:
        sanswer = main_split.proccess(question)
        answer = json.loads(sanswer)
    except:
        answer = {}
        pass
    result = {}
    if "useful" in answer:
        result["APIres"] = answer["useful"]
    else:
        result["APIres"] = False
    if "decision" in answer:
        result["decided"] = True
    else:
        result["decided"] = False
    return result, answer

if __name__ == "__main__":
    with open(sys.argv[1]) as quescsv:
        parsecsv(quescsv)