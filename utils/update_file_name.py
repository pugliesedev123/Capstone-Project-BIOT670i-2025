import os, shutil, re

Print("Input an owner folder as the top folder and script will iterate through the folder to rename files using the taxon name")


##you can either hard code the path or input it
#path_to_owner = ''

#or have user input path
path_to_owner = input("Paste path: ")

#Move to inputted filepath
os.chdir(path_to_owner)

#Get basic folder name
owner = os.path.basename(path_to_owner)
#initialize a file to track name changes
with open(f"{owner}_file_changes.csv", "w") as f:
    f.write("oldName,newName\n")

    #Loop through folders in directory
    for folder in os.listdir():
        if os.path.isdir(folder):
            ##Replace spaces in folder name with underscore
            newName = folder.replace(" ","_")
            os.rename(folder, newName)
            
            ##Go into each folder and rename all files
            os.chdir(newName)
            #Get number of files/images within folder
            num = len(os.listdir())

            #convert all files to jpg
            #Can change x to a different number if you want to start at a higher number
            x = 1
            #Empty dictionary to track what the original file was, the temporary file name, and final file name
            track_name = {}
            #Loop through the taxon folder contents
            for file in os.listdir():
                if os.path.isfile(file) == True and ".DS_Store" not in file:
                    #rename as generic name to avoid double naming
                    newFile = f"image ({x}).jpg"
                    os.rename(file, newFile)
                    x+=1
                    #store new name as key and old name as list item in value
                    track_name[newFile]=[file]

            #Loop through the taxon folder contents again to do final file name revisions
            for file in os.listdir():
                if os.path.isfile(file) == True and ".DS_Store" not in file:
                    #print(f"This file: {track_name.get(file)}")
                    #Get taxon name from folder name "newName"
                    taxon = newName.replace("taxon-","")
                    #Replace generic text with taxon
                    updatedFile = file.replace("image", taxon)
                    os.rename(file, updatedFile)
                    #Write to output file the name changes
                    f.write(f"{owner}\\{folder}\\{file},{owner}\\{newName}\\{updatedFile}\n")

            #Move back into owner folder
            os.chdir(path_to_owner)
