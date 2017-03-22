# -*- coding: utf-8 -*-
"""
Replace blank columns representing the fact that the restaurant aspect is 
not mentioned with '-'.
"""

import csv

def write_vals(reader, writer):
    for row in reader:
        writer.writerow(row)
        break
    
    for row in reader:
        for i in range(10, 13):
            if row[i] == '':
                row[i] = '-'
        writer.writerow(row)

    
    
if __name__ == "__main__":
    fname = "labeledChineseReview"
    ofname = "labeledReview"
    
    for i in range(1,6):
        filename = fname+str(i)+".csv"
        ofilename = ofname+str(i)+".csv"
        
        f = open(filename, 'r', encoding = 'utf8')
        reader = csv.reader(f)
        
        of = open(ofilename, 'w', newline = '', encoding = 'utf8')
        writer = csv.writer(of)
        
        write_vals(reader, writer)
        f.close()
        of.close()