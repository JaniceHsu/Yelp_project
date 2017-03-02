# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 00:52:15 2017
Get reviews given a business
@author: Janice
"""
import csv

def get_reviews(_id, n):
    
    #fname = "../dataset-examples-master/yelp_academic_dataset_review.csv"
    fname = "reviews_NEGallUSAAmericanRes.csv" 
    f = open(fname, 'r', encoding = 'utf8')
    reader = csv.reader(f)
    
    ofname = "reviews_allUSA" + n + "Res.csv"
    of = open(ofname, 'w', newline = '', encoding = 'utf8')
    writer = csv.writer(of)
    
    _ofname = "reviews_NEGallUSA" + n + "Res.csv"
    _of  = open(_ofname, 'w', newline = '', encoding = 'utf8')
    _writer = csv.writer(_of)
    
    
    
    x = 0
    
    for row in reader:
        writer.writerow(row)
        _writer.writerow(row)
        break
    
    for row in reader:
        x+=1
        if row[4] in _id:
            writer.writerow(row)
        else:
            _writer.writerow(row)
        if x%100000 == 0:
            print(x)
    
    _of.close()
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
        except:
            pass
    return ids
    
if __name__ == "__main__":    
    n = "Indian"
    fname = "business_allUSA" + n + "Res.csv"
    f = open(fname, 'r', encoding = 'utf8')
    reader = csv.reader(f)
    _id = get_ids(reader)
    print(_id[:20])
    

    get_reviews(_id, n)
    
    f.close()
    print(n)
    print("END")
        
