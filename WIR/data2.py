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
with open(filename, 'r',encoding="utf8") as csvfile: 
	# creating a csv reader object 
	csvreader = csv.reader(csvfile) 
	
	# extracting field names through first row 
	#fields = csvreader.next() 

	# extracting each data row one by one
	
	for row in csvreader: 
		rows.append(row)
		

	# get total number of rows 
	print("Total no. of rows: %d"%(csvreader.line_num)) 

rows.pop(0)

for row in rows:
        i=0
        value = []
        for col in row:
                col=col.translate(non_bmp_map)
                print(col)
                if col == '': col='0'
                value.append(col)
        print('\n')
        print('\n')



        print(value)



        

        


			
				

