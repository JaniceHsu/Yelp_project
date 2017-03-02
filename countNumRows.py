# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:48:42 2017

@author: Janice
"""


def countRows(fname):

    fname = "reviews_NEGallUSA" + n + "Res.csv"

    f = open(fname, 'r', encoding = 'utf8')
    
    reader = csv.reader(f)
    x = -1
    for row in reader:
        x+=1
    print(fname, ":", x , " lines total", sep = '')

    f.close()  
    
if __name__ == "__main__":
    print("START")

    print("END")