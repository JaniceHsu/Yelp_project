# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:48:42 2017

@author: Janice
"""
import csv

def countRows(fname):
    print("CountRows")
    #fname = "reviews_NEGallUSA" + n + "Res.csv"

    f = open(fname, 'r', encoding = 'utf8')
    
    reader = csv.reader(f)
    x = -1
    for row in reader:
        x+=1

    f.close()  
    return x
    
if __name__ == "__main__":
    print("START")
    files = ['reviews_allUSAAmericanRes.csv',
    'reviews_allUSAChineseRes.csv',
    'reviews_allUSAIndianRes.csv',
    'reviews_allUSAItalianRes.csv',
    'reviews_allUSAJapaneseRes.csv' ,
    'reviews_allUSAMexicanRes.csv',
    'reviews_NEGallUSASearchedRes.csv']
    for i in files:
        x =         countRows(i)
        print(i,":", x , " lines total", sep = '') 

    print("END")