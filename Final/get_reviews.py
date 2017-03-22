# -*- coding: utf-8 -*-
"""
Get Yelp reviews given business.
Input:
    CSV file of Yelp business.
Output:
    CSV file of Yelp reviews on businesses from the input file.
@author: Janice
"""
import csv

def get_reviews(_id, n, x):
    
    fname = "reviews_allUSAChineseRes.csv"
    f = open(fname, 'r', encoding = 'utf8')
    reader = csv.reader(f)
    
    ofname = "selectChineseReview"+str(x)+".csv"
    of = open(ofname, 'w', newline = '', encoding = 'utf8')
    writer = csv.writer(of)  
    
    x = 0
    
    for row in reader:
        writer.writerow(row)
        break
    
    for row in reader:
        x+=1
        if row[4] in _id:
            writer.writerow(row)
        if x%100000 == 0:
            print(x)
    of.close()
    
def get_ids(reader):
    ids = []
    for row in reader:
        try:
            if reader.line_num == 1:
                print(list(enumerate(row)))
                continue
            else:
                ids.append(row[1])
                print(row[9])
        except:
            pass
    return ids
    
if __name__ == "__main__":    
    n = "Chinese"
    fname = "select_ChineseRes.csv"
    f = open(fname, 'r', encoding = 'utf8')
    reader = csv.reader(f)
    _id = get_ids(reader)
    print(_id[:20])    
    f.close()
    print(n)
    print("END")
        
