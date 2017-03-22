# -*- coding: utf-8 -*-
"""
Used to preprocess the data.
Reads a CSV file containing yelp reviews and associated information (user_id, date, text, stars, etc.)
Writes a CSV file containing the same information plus a sentiment label.
Adds an attribute 'sentiment' to the CSV file. 
Adds the value 1 when the attribute 'stars' contains the value 4 or 5 to represent a positive sentiment.
Adds the value 0 when the attribute 'stars' contasin the value 1 or 2 to represent a negative sentiment. 

"""
import csv
    
def categorize(reader, d, x, i):
    of = "reviews_" + n + "PN.csv"
    ofile  = open(of, 'w', newline = '', encoding = 'utf8')
    writer = csv.writer(ofile)
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
          
            if int(row[5]) > 3:
                writer.writerow([row[3], 1])
                pos += 1
            elif int(row[5]) <3:
                writer.writerow([row[3], 0])
                neg += 1
        except Exception as e: 
            print("-----")
            print(str(e))
            print(row[3])
    # Prints the number of positive/negative reviews
    # Ensure that there is a fair amount of positive/negative. (not all positive or all negative)
    print("Pos:", pos)
    print("Neg:", neg)
    ofile.close()

if __name__ == "__main__":    
    n = "Chinese"
    fname = "reviews_allUSA" + n + "Res.csv"
    f = open(fname, 'r', encoding = 'utf8')
    reader = csv.reader(f)
    
    d = dict()
    categorize(reader, d, 0, 5)

    d = sorted(d.items())#,key = lambda x: (x[1], x[0]), reverse = True)
    for i in d:
        print(i)
        
    f.close()
    print(n)
    print("END")
        