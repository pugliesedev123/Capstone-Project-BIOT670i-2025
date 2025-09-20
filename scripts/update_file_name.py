import os, shutil, re

##input an owner folder as the parent folder and script will iterate through the folder to rename files not of the correct naming convention


##you can either hard code the path or input it
path_to_owner = 'path/to/folder'

#or have user input path
#path_to_owner = input("Paste path: ")


os.chdir(path_to_owner)

owner = os.path.basename(path_to_owner)
#initialize a file to track name changes
with open(f"{owner}_file_changes.csv", "w") as f:
    f.write("oldName,newName\n")

    for folder in os.listdir():
        if os.path.isdir(folder):
            print(folder)

            ##Replace spaces in folder name with underscore
            newName = folder.replace(" ","_")
            os.rename(folder, newName)
            print(newName)

            

            ##Go into each folder and updated files that do not follow naming convention, skip files that already are of correct convention
            os.chdir(newName)
            
            num = len(os.listdir())
            #Make list of possible names
            poss_files = []
            for x in range(num+2):
                poss_files.append(f"{newName.replace('taxon-',"")} ({x+1}).jpg")
            avail_files = poss_files.copy()
            for file in os.listdir():
                
                if os.path.isfile(file) == True and ".DS_Store" not in file:
                    print(f"This file: {file}")
                    ##Fix file if semi-correct
                    p = re.compile(r"([\s\S]*) \(\d+\)")
                    m = re.search(p,file)
                    newFile = ""
                    if m != None:
                        print(m.group(1))
                        newFile = file.replace(m.group(1),m.group(1).replace(" ","_"))
                    
                    #if File name already fits the naming convention, skip
                    if file in poss_files:
                        #remove from list of possible file names for other files in the folder
                        print(f"{file} is named correctly. skip.")
                        break
                    #if the File name doesn't fit convention
                    elif newFile in poss_files:
                        print(f"changing {file} to {newFile}")
                        os.rename(file, newFile)
                        f.write(f"{file},{newFile}\n")

                    else:
                        ##first see if name is just missing _ in taxon name
            
                            #Get first available file name
                            newFile = avail_files[0]
                            print(f"try changing to {newFile}")
                            #if the File name is already used for some other file in the folder loop through possible file names
                            while newFile in os.listdir():
                                print(f"{newFile} already exists folder. choose new name.")
                                avail_files.remove(newFile)
                                newFile = avail_files[0]
                                print(f"try this file {newFile}")
                            
                            avail_files.remove(newFile)
                            print(f"changing {file} to {newFile}")
                            os.rename(file, newFile)
                            f.write(f"{file},{newFile}\n")


                    if ".DS_Store" in file:
                        print("deleting "+file)

            os.chdir(path_to_owner)

