import os, sys


indir = sys.argv[1]
outdir= sys.argv[2]

files = os.listdir(indir)

for f in files:
    newname = os.path.join(outdir, f.split(".")[0] + "_masks.tif")
    oldname = os.path.join(indir, f)
    print(oldname, newname)
    
    os.rename(oldname, newname) 
