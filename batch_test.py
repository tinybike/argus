import csv
import sys
from main_frame import get_answer



TSVFILE=sys.argv[1]

for line in csv.reader(open(TSVFILE), delimiter=',',skipinitialspace=True):
    print "QID=",line[0],", Question=",line[1],", Answer=",get_answer(line[1])
