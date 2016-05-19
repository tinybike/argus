import urllib.request
import sys
import json

def makequery(question):
    print(question)


if __name__ == "__main__":
    with open(sys.argv[1]) as que:
        question = json.load(que)
        print(makequery(question))
