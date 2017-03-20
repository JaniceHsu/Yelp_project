# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 00:52:15 2017
Get reviews given a business
@author: Janice
"""
import csv

def get_reviews(_id, n, x):
    
    #fname = "../dataset-examples-master/yelp_academic_dataset_review.csv"
#    fname = "reviews_NEGallUSAAmericanRes.csv" 
    fname = "reviews_allUSAChineseRes.csv"
    f = open(fname, 'r', encoding = 'utf8')
    reader = csv.reader(f)
    
    ofname = "selectChineseReview"+str(x)+".csv"
    of = open(ofname, 'w', newline = '', encoding = 'utf8')
    writer = csv.writer(of)
    
#    _ofname = "reviews_NEGallUSA" + n + "Res.csv"
#    _of  = open(_ofname, 'w', newline = '', encoding = 'utf8')
#    _writer = csv.writer(_of)
    
    
    
    x = 0
    
    for row in reader:
        writer.writerow(row)
#        _writer.writerow(row)
        break
    
    for row in reader:
        x+=1
        if row[4] in _id:
            writer.writerow(row)
#        else:
#            _writer.writerow(row)
        if x%100000 == 0:
            print(x)
    
    #_of.close()
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
    #n = "Indian"
    n = "Chinese"
    #fname = "business_allUSA" + n + "Res.csv"
    fname = "select_ChineseRes.csv"
    f = open(fname, 'r', encoding = 'utf8')
    reader = csv.reader(f)
    _id = get_ids(reader)
    print(_id[:20])
    
    #for i in range(len(_id)):
    #    get_reviews(_id[i], n, i+1)
    
    f.close()
    print(n)
    print("END")
        
