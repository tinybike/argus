import sys
import json
import xml.etree.ElementTree as ET

try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen

def makequery(question):

    if question["type"] != "weather":
        print("Doesn't look like a crypto_currency query to me, can't do.")
        answer = {"useful": False}
        return answer
    else:
        source = "wolfram_alpha_api"
        # TODO find better API google
        response_xml = urlopen("http://api.wolframalpha.com/v2/query?input=weather+Kiev+from+18.05.1990+to+18.05.2001&appid=URY5GL-8AV552TY5W").read()

        root = ET.fromstring(response_xml)

     #   answer["useful"] = True
     #   answer["source"] = source
   # return answer

if __name__ == "__main__":
    with open(sys.argv[1]) as que:
        question = json.load(que)
        print(makequery(question))