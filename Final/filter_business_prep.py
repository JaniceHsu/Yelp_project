
"""
Get statistics on yelp business reviews to decide how to filter businesses.

Input:
    yelp_academic_dataset_business.csv 
    (contains all yelp businesses in the downloaded dataset and their attributes)
Output:
    Information on a given business attribute. 
    E.g. business categories (value, number_of_occurences)
        ('Restaurants', 48485)
        ('Shopping', 22466)
        ('Food', 21189)
        ('Beauty & Spas', 13711)
        ('Home Services', 11241)
        ('Nightlife', 10524)
        ('Health & Medical', 10476)
        ('Bars', 9087)
        ('Automotive', 8554)
        ('Local Services', 8133)
        ...
        ('Alsatian', 1)
        ('Agriturismi', 1)
    
    
E.g. stats on restaurant categories
Yelp business attributes:
neighborhood
business_id
hours
is_open
address
attributes
categories
city
review_count
name
longitude
state
stars
latitude
postal_code
type
"""

import csv

def get_attrib(reader, d, x):
    for row in reader:
        try:
            if reader.line_num == 1:
                print(list(enumerate(row)))
                continue
            x +=1
            if x %2000 ==0 :
                print(x)
            if row[6] != "":
                attribs= eval(row[6])
                if 'Restaurants' in attribs: 
                    for i in attribs: 
                        if i not in d.keys():
                            d[i] = 1
                        else:
                            d[i] +=1
        except:
            print(reader.line_num, row[9])
            
def get_cat(reader, d, x):
    no_cat = 0
    for row in reader:
        try:
            if reader.line_num == 1:
                print(list(enumerate(row)))
                continue
            x +=1
            if x %2000 ==0 :
                print(x)
            
            if row[6] != "":
                cats= eval(row[6])
                for j in cats:        
                    #if i.startswith("RestaurantsPriceRange2"):
                    #j = i[-1:]
                    if j not in d.keys():
                        d[j] = 1
                    else:
                        d[j] +=1
            else:
                no_cat += 1
        except:
            print(reader.line_num, row[9]) 
        
def test_attrib_combos(reader):
    f = dict()
    r = dict()
    d = dict()
    for row in reader:
        if row[6] != "":
            attrib = eval(row[6])
            if 'Food' in attrib and 'Restaurants' not in attrib:
                if row[9] not in f:
                    f[row[9]] = 1
                else:
                    f[row[9]] +=1
            elif 'Food' not in attrib and 'Restaurants'in attrib:
                if row[9] not in r:
                    r[row[9]] = 1
                else:
                    r[row[9]] +=1
            elif 'Food' in attrib and 'Restaurants' in attrib:
                if row[9] not in d:
                    d[row[9]] = 1
                else:
                    d[row[9]] +=1  
    print("food w/o restaurants")
    print_dict(f)
    print("restaurants w/o food")
    print_dict(r)
    print("food and restaurants")
    print_dict(d) 
    
def store_state_abbrev():
    fname = "state_abbrev.txt"
    f = open(fname, 'r')
    state_list = f.read().splitlines()
    f.close()
    return state_list

def get_state(reader, d, x):
    for row in reader:
        try:
            if reader.line_num == 1:
                print(list(enumerate(row)))
                continue
            x +=1
            if x %2000 ==0 :
                print(x)
            info = row[11]
            if info not in d.keys():
                d[info] = 1
            else:
                d[info] +=1
        except:
            print(reader.line_num, row[9])

def eval_state(d):            
    state_list = store_state_abbrev()
    us_state = []
    not_us_state = []
    for i in d.keys():
        if i in state_list:
            us_state.append(i)
        else:
            not_us_state.append(i)
    print("us_state", us_state)
    print("not_us_state", not_us_state)

def get_general(reader, d, x, i):
    for row in reader:
        try:
            if reader.line_num == 1:
                print(list(enumerate(row)))
                continue
            x +=1
            if x %2000 ==0 :
                print(x)
            info = row[i]
            if info not in d.keys():
                d[info] = 1
            else:
                d[info] +=1
        except:
            print(reader.line_num, row[9])
    print(x)       

      
def print_dict(d):
    d = sorted(d.items(),key = lambda x: (x[1], x[0]), reverse = True)
    for i in d[:10]:
        print(i)
    print("=")
    for i in d[-10:]:
        print(i)  
    print("===========")
    return


            
        
def test_combos(reader):
    f = dict()
    r = dict()
    d = dict()
    for row in reader:
        if row[6] != "":
            attrib = eval(row[6])
            if 'Food' in attrib and 'Restaurants' not in attrib:
                if row[9] not in f:
                    f[row[9]] = 1
                else:
                    f[row[9]] +=1
            elif 'Food' not in attrib and 'Restaurants'in attrib:
                if row[9] not in r:
                    r[row[9]] = 1
                else:
                    r[row[9]] +=1
            elif 'Food' in attrib and 'Restaurants' in attrib:
                if row[9] not in d:
                    d[row[9]] = 1
                else:
                    d[row[9]] +=1  
    print("food w/o restaurants------")
    print_dict(f)
    print("restaurants w/o food------")
    print_dict(r)
    print("food and restaurants------")
    print_dict(d) 
    f = set(f.keys())
    r = set(r.keys())
    d = set(d.keys())
    fr = [f,r]
    fd = [f,d]
    rd = [r,d]
    frd = [f,r,d]

    oall = set.intersection(*frd)
    print("frd: -------")
    print(oall)  
    
    print("fr: -------")
    print(set.intersection(*fr)-oall)
    print("fd: -------")
    print(set.intersection(*fd)-oall)
    print("rd: -------")
    print(set.intersection(*rd)-oall)
  
        
      
#['neighborhood', 'business_id', 'hours', 'is_open', 'address', 'attributes', 'categories', 'city', 'review_count', 'name', 'longitude', 'state', 'stars', 'latitude', 'postal_code', 'type']
#[(0, 'neighborhood'), (1, 'business_id'), (2, 'hours'), (3, 'is_open'), 
#(4, 'address'), (5, 'attributes'), (6, 'categories'), (7, 'city'), 
#(8, 'review_count'), (9, 'name'), (10, 'longitude'), (11, 'state'), 
#(12, 'stars'), (13, 'latitude'), (14, 'postal_code'), (15, 'type')]
    
if __name__ == "__main__":
    print("START")
    fname = "../dataset-examples-master/yelp_academic_dataset_business.csv"
    f = open(fname, 'r', encoding = 'utf8')
    
    reader = csv.reader(f)

    x = 0
    d = dict()
    
    #get_attrib(reader, d, x)
    get_cat(reader, d, x)
    #get_general(reader, d, x, 7)
    #get_state(reader, d, x)
    #eval_state(d)
    

    d = sorted(d.items(),key = lambda x: (x[1], x[0]), reverse = True)
    for i in d[:10]:
        print(i)
    print()
    for i in d[-10:]:
        print(i)
    print(len(d))

    f.close()
    # of.close()
    print("END")
    