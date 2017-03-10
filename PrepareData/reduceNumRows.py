# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 20:12:45 2017

@author: Janice
"""
import csv
import random
def reduce(fname, total):
    print("CountRows")
    #fname = "reviews_NEGallUSA" + n + "Res.csv"

    f = open(fname, 'r', encoding = 'utf8')
    
    nfname = fname[:-4]+"_"+str(total)+".csv"
    owr = open(nfname, 'w', newline = '', encoding = 'utf8')
    writer = csv.writer(owr)
    reader = csv.reader(f)
    
    for row in reader:
        writer.writerow(row)
        break
    x = 0
    for row in reader:
        if random.random()>.8 and x < total:
            writer.writerow(row)
            x+=1

    f.close()  
    owr.close()
    return x
if __name__ == "__main__":
    print(reduce("reviews_ChinesePN.csv", 10000))