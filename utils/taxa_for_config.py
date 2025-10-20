import glob
import os
import re
from pathlib import Path

##Creates two files containing the taxa within data/train and which files/folders may have been skipped

def write_to_file(file, text):

    header = "The following is a list of valid taxa categories that exist currently in the database.\nReplace the + with a - if you wish to exclude this taxon.\n\nIf running your classification with the argument '--include-config-classes-only', delete all rows from the generated .txt besides those you wish to include that start with '+'\n"
    #Write 'text', which is a list, to appropriate file
    with open(file, "w") as f:
        f.write(f"{header}")
        for item in text:
            f.write(f"+{item}\n")

def main():
    #Initialize empty list
    taxa = []

    #Recursively go through data/train folder to get all folders
    for item in glob.glob(r"./data/train/**/*", recursive=True):
        #Match for taxon folder
        #Grab match and add to list
        name = Path(item).name
        if name.startswith("taxon-"):
            taxa.append(name[len("taxon-"):].lower())

    #Remove duplicates and sort
    unique_taxons = list(set(taxa))
    unique_taxons.sort()

    #Write lists to files
    write_to_file("taxa-config.txt",unique_taxons)
    
if __name__ == "__main__":
    main()