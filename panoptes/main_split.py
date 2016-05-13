import sys
from finanget import stockquery

def proccess(que):
    ans = stockquery(que)
    if "Decision" in ans:
        return ans

if __name__ == "__main__":
    with open(sys.argv[1]) as que:
        print(proccess(que))
