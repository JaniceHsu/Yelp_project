# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 00:17:44 2017

@author: Janice
"""
import csv

fname = "selectChineseReview1.csv"

reader = csv.reader(f)

f = open(fname, 'r', encoding = 'utf8')

reader = csv.reader(f)

for row in reader:
    print(row[1], row[10:13])