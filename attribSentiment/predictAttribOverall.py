# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 22:41:49 2017

@author: Janice
"""

import csv
from variousLearnersOverall import *
import nltk


def separateTextTarget(reader):
    food = []
    price = []
    service = []
    for row in reader:
        break
    for row in reader:
        food.append((row[3], row[10]))
        price.append((row[3], row[11]))
        service.append((row[3], row[12]))

    return (food, price, service)
    
def separateNoStopWards(reader):
    