import csv
import json
import sys
import sys

if __name__ == "__main__":



    with open(sys.argv[1]) as quescsv:
        stuff = {}
        questions = csv.reader(quescsv)
        for row in questions:
            for rowy in row:
                if "{" in rowy:
                    ajson=json.loads(rowy)
                    print(ajson)