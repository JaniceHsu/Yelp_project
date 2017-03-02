# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 18:54:02 2017

@author: Janice


"""
import csv
import re
    
def categorize2(reader, d, x, i):
    of = "reviews_" + n + "PN.csv"
    ofile  = open(of, 'w', newline = '', encoding = 'utf8')
    writer = csv.writer(ofile)

#    fp = open("reviews_" + n + "_Pos.txt", 'w')
#    fn = open("reviews_" + n + "_Neg.txt", 'w')

    pos = 0
    neg = 0

    for row in reader:
        try:
            if reader.line_num == 1:
                print(list(enumerate(row)))
                writer.writerow([row[3], 'sentiment'])
                continue
            x +=1
            if x % 20000 == 0:
                print(x)
          
            if int(row[5]) > 2:
                writer.writerow([row[3], 1])
                #fp.write("{}\n".format(re.sub("\n", " ", row[3])))
                pos += 1
            else:
                writer.writerow([row[3], 0])
                #fn.write("{}\n".format(re.sub("\n", " ", row[3])))
                neg += 1
        except Exception as e: 
            print("-----")
            print(str(e))
            print(row[3])#reader.line_num, str(e), row[5], row[3])
    print("Pos:", pos)
    print("Neg:", neg)
#    fp.close()
#    fn.close()
    ofile.close()

def get_general(reader, d, x, i):
    ofpos = "reviews_" + n + "_Pos.csv"
    ofp  = open(ofpos, 'w', newline = '', encoding = 'utf8')
    pos_writer = csv.writer(ofp)

    ofneg = "reviews_" + n + "_Neg.csv"
    ofn  = open(ofneg, 'w', newline = '', encoding = 'utf8')
    neg_writer = csv.writer(ofn)
    
    fp = open("reviews_" + n + "_Pos.txt", 'w')
    fn = open("reviews_" + n + "_Neg.txt", 'w')
    
    pos = 0
    neg = 0
#    for row in reader:
#        pos_writer.write(row[3])
#        neg_writer.write(row[3])
#        break
    
    for row in reader:
        try:
            if reader.line_num == 1:
                print(list(enumerate(row)))
                continue
            x +=1
            if x > 10:# 20000 == 0:
                print(type(x))
                break
                print(x)
            print("A")
            
            if int(row[5]) > 2:
                pos_writer.writerow([row[3], 1])
                fp.write(row[3])
                pos += 1
                print("B")                
            else:
                neg_writer.writerow([row[3], 0])
                fn.write(row[3])
                neg += 1
                print("C")
        except:
            print(reader.line_num, row[5])
    print("Pos:", pos)
    print("Neg:", neg)
    fp.close()
    fn.close()
    ofp.close()
    ofn.close()
      
def print_dict(d):
    d = sorted(d.items(),key = lambda x: (x[1], x[0]), reverse = True)
    for i in d[:10]:
        print(i)
    print("=")
    for i in d[-10:]:
        print(i)  
    print("===========")
    return


            
    

if __name__ == "__main__":    
    n = "Chinese"
    fname = "reviews_allUSA" + n + "Res.csv"
    f = open(fname, 'r', encoding = 'utf8')
    reader = csv.reader(f)
    
    d = dict()
    
    categorize2(reader, d, 0, 5)
    

    d = sorted(d.items())#,key = lambda x: (x[1], x[0]), reverse = True)
    for i in d:
        print(i)
        
    f.close()
    print(n)
    print("END")
        