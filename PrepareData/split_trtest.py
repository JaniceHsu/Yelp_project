# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 05:29:19 2017

@author: Janice
"""

import countNumRows
import csv
import random

def create2files(fname, total):
    trainNum = int(total*.8)
    testNum = total-trainNum
    
    testfname = fname[:-4]+"_test.csv"
    trainfname = fname[:-4]+"_train.csv"
    
    ote = open(testfname, 'w', newline = '', encoding = 'utf8')
    otr = open(trainfname, 'w', newline = '', encoding = 'utf8')
    
    wtest = csv.writer(ote)
    wtrain =csv.writer(otr)
    
    f = open(fname, 'r', encoding = 'utf8')
    reader = csv.reader(f)
    
    trN = 0
    teN = 0
    for row in reader:
        wtest.writerow(row)
        wtrain.writerow(row)
        break
    for row in reader:
        if teN == trainNum:
            wtrain.writerow(row)
            trN+=1
        elif trN == trainNum:
            wtest.writerow(row)
            teN+=1
        elif (random.randint(0,5)) == 0:
            wtest.writerow(row)
            teN+= 1
        else:
            wtrain.writerow(row)
            trN += 1
    ote.close()
    otr.close()
    f.close()
    print(teN, testNum)
    print(trN, trainNum)
    
    
    
if __name__ == "__main__":
    fname = "reviews_ChinesePN.csv"
    total = countNumRows.countRows(fname)

    create2files(fname, total)
    

    