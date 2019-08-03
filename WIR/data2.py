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

        countURL=value[1].count("https")
        value.append(countURL)

        countHashtag=value[1].count("#")
        value.append(countHashtag)

        countMensions=value[1].count("@")
        value.append(countMensions)

        averageHashtag=countHashtag/len(value[1].split()) #rispetto le parole della frase
        averageURL=countURL/len(value[1].split())

        value.append(averageHashtag)
        value.append(averageURL)
        value.append(len(value[1].split()))
        print(value)



        

        


			
				

