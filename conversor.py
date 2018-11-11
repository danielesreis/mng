import subprocess
import os

path = os.getcwd()
# dest = 
files = os.listdir(path)

# raw2jpg.exe -i <input file> -o <output folder>
# you must run it from C:\Program Files (x86)\Easy2Convert Software\RAW to JPG
# from http://www.easy2convert.com/raw2jpg/
for file in files:
	subprocess.run(["raw2jpg.exe", "-i", file, "-o", dest])
