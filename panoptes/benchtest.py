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
                try:
                    alpha = rowy[0].isalpha()
                    space = (rowy[0] != '\n')
                except:
                    pass
                if not alpha and space:
                    print(rowy)
