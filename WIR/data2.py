import os
import shutil
import sys  ########

# importing csv module 
import csv 

# csv file name 
filename = "test.csv"

# initializing the titles and rows list 
fields = [] 
rows = [] 

non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
# reading csv file 
with open(filename, 'r') as csvfile: 
	# creating a csv reader object 
	csvreader = csv.reader(csvfile) 
	
	# extracting field names through first row 
	#fields = csvreader.next() 

	# extracting each data row one by one 
	for row in csvreader: 
		rows.append(row) 

	# get total number of rows 
	print("Total no. of rows: %d"%(csvreader.line_num)) 


for row in rows:
        i=0
        value = []
        for col1 in row:
                col=col1.translate(non_bmp_map)
                print(col)
                value.append(col)
        print('\n')
        print('\n')
	#value = row.split(",")
        while i<len(value)-1:
                if (len(value)==1):break
                if isinstance(value[i],str) and isinstance(value[i+1],str) and not value[i].isdigit() and not value[i+1].isdigit() :
                    value[i]=value[i]+","+value[i+1]
                    value.remove(value[i+1])
                    i=0
                    continue
                i=i+1

        print(value)

        


			
				

