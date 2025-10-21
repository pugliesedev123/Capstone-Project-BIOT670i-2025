import os, re

print("This script will output a csv file with two columns: a directory column and a file name column")
print("This script will traverse down 3 levels i.e. Top folder > Subfolder (owner) > Sub-subfolder (taxon) to get the contents of the sub-subfolder")
print("It will also remove empty taxon folders from the existing owner\\* subfolders")


if os.name = "posix":
    #For Mac
    user_path = "data/train"
    pattern = re.compile(r"[\s\S]*/data/train/")
    pattern2 = "/"

else:
    #For Windows
    user_path = "data\\train"
    pattern = re.compile(r"[\s\S]*\\data\\train\\")
    pattern2 = "\\"


output = "Directory_File_output.csv"

#Comment out the above and uncomment the below to run as user decided
#paste the file path
#user_path = input("Paste path of directory to get contents")
'''
if os.name = "posix":
    pattern2 = "/"
else:
    pattern2 = "\\"

pattern = user_path[:user_path.rfind(pattern2)+1]
'''


#Change to user path 
os.chdir(user_path)


#Keep output file open throughout directory search
with open(output, "w") as o:
    #Loop through contents of the Top Folder (user_path)
    for folder in os.listdir():
        #Move into subfolder if directory
        if os.path.isdir(folder):
            os.chdir(folder)
            #print(f"in {folder}")
            #Loop through contents of subfolder
            for f2 in os.listdir():
                #Move into sub-subfolder f2 if directory
                if os.path.isdir(f2):
                    os.chdir(f2)
                    #Check if folder is empty. If empty, delete folder.
                    if len(os.listdir()) == 0:
                        print(f"Remove {f2} from data/{folder}!")
                        os.chdir("..")
                        os.rmdir(f2)
                        continue
                    #Process the filepath text in order to get the owner/taxon file path only
                    directory = f"{folder}{pattern2}{f2}"
                    #Get all of the files in the current taxon folder
                    for file in os.listdir(): 
                        #DS_Store conditional was because macbook has these weird hidden files that keep popping up
                        if os.path.isfile(file) and "DS_Store" not in file:
                            #Write the directory and file to the output file
                            o.write(directory + "," + file + "\n")

                    #After finishing reading files in taxon folder, move back up to the owner folder
                    os.chdir("..")
            #After finishing the taxon folders in an owner folder, move back up to top folder
            os.chdir("..")

print("Script Completed")
