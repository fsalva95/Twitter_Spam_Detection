import os
import shutil
import sys  ########
PATH="/home/biar/Desktop/WIR/progettoGIT/WIR"  #da modificare per il vostro path


non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
with open(PATH+"\\test.csv","r", encoding="utf8") as f:
        for r in f:
            t=r.translate(non_bmp_map)
            print(t)
            value=t.split(",")
            i=0
            while i < len(value)-1:
                if (len(value)==1): break
                if isinstance(value[i], str) and isinstance(value[i+1], str) and not value[i].isdigit() and not value[i+1].isdigit() :
                    value[i]=value[i]+","+value[i+1]
                    value.remove(value[i+1])
                    i=0
                    continue
                i=i+1

            value = [w.replace('\n', '') for w in value]
            value = [w.replace('\"', '') for w in value]
            print(value)
			
				

