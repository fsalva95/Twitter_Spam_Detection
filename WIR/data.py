import os
import shutil
import sys  ########
PATH="C:\\Users\\Salvatore\\Desktop\\WIR\\"  #da modificare per il vostro path


non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
with open(PATH+"\\test.csv","r", encoding="utf8") as f:
        for r in f:
                print(r.translate(non_bmp_map))

