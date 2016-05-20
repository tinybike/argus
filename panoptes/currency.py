import urllib.request
import sys
import json

def makequery(question):

    if question["type"] != "currency":
        print("Doesn't look like a currency query to me, can't do.")
        answer = {"useful": False}
        return answer

    res = urllib.request.urlopen("http://api.fixer.io/"+question["datestart"])
    response = json.loads(str(res))
    print(response)

if __name__ == "__main__":
    with open(sys.argv[1]) as que:
        question = json.load(que)
        print(makequery(question))
