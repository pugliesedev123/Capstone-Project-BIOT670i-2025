import os, re

print("This script will output a csv file with two columns: a directory column and a file name column")
print("It will also remove empty taxon folders from the existing owner\\* subfolders")

#For Mac
path = "data/train"
pattern = re.compile(r"[\s\S]*/data/train/")
output = "Directory_File_output.csv"

#For Windows
#path = "data\\train"
#pattern = re.compile(r"[\s\S]*\\data\\train\\")
#output = "data\\Folders_output.txt"

os.chdir(path)

with open(output, "w") as o:
    for folder in os.listdir():
        if os.path.isdir(folder):
            os.chdir(folder)
            #print(f"in {folder}")
            for f2 in os.listdir():
                if os.path.isdir(f2):
                    os.chdir(f2)
                    #print(f"in {f2}")
                    if len(os.listdir()) == 0:
                        print(f"Remove {f2} from data/{folder}!")
                        os.chdir("..")
                        os.rmdir(f2)
                        continue
                    fullpath = os.path.abspath(folder)
                    remove = re.search(pattern,fullpath).group()
                    #print(fullpath)
                    fullpath = fullpath.replace(remove, "")
                    #print(fullpath)
                    directory = fullpath[:fullpath.rfind("/")]
                    for file in os.listdir(): 
                        #DS_Store conditional was because my macbook has these weird hidden files that keep popping up
                        if os.path.isfile(file) and "DS_Store" not in file:
                            #print(file)
                            o.write(directory + "," + file + "\n")

                        
                    os.chdir("..")
            os.chdir("..")

print("Script Completed")