import os, shutil, re

##input an owner folder as the parent folder and script will iterate through the folder to rename files using the taxon name


##you can either hard code the path or input it
#path_to_owner = ''

#or have user input path
path_to_owner = input("Paste path: ")


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

            

            ##Go into each folder and rename all files
            os.chdir(newName)
        
            num = len(os.listdir())

            #convert all files to jpg
            #change x to a different number if you want to start at a higher number since files already exist in folder
            x = 1
            track_name = {}
            for file in os.listdir():
                if os.path.isfile(file) == True and ".DS_Store" not in file:
                    newFile = f"image ({x}).jpg"
                    os.rename(file, newFile)
                    x+=1
                    track_name[newFile]=[file]
                
            for file in os.listdir():
                if os.path.isfile(file) == True and ".DS_Store" not in file:

                    print(f"This file: {track_name.get(file)}")
                    taxon = newName.replace("taxon-","")
                    updatedFile = file.replace("image", taxon)
                    os.rename(file, updatedFile)
                    f.write(f"{owner}\\{folder}\\{file},{owner}\\{newName}\\{updatedFile}\n")

           
            os.chdir(path_to_owner)