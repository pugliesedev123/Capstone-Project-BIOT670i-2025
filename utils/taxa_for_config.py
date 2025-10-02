import glob
import os
import re


##Creates two files containing the taxa within data/train and which files/folders may have been skipped

def write_to_file(file, text):
    #Decide header of file based on filename
    if file == "config.txt":
        header= "Taxon"
    elif file == "ignore.txt":
        header = "Ignored items in directory"
    #Write 'text', which is a list, to appropriate file
    with open(file, "w") as f:
        f.write(f"{header},\n")
        for item in text:
            f.write(f"{item},\n")

def get_taxa():

    #Initialize empty lists
    taxa = []
    ignore = []
    #Pattern to get taxon from folder
    pattern = re.compile(r"taxon-(\S*)")
    #Recursively go through data/train folder to get all folders
    for item in glob.glob(r"data//train//*//*", recursive=True):
        #Match the pattern for taxon
        match = re.search(pattern, item)
        #Skip if no match, but log for troubleshooting
        if match == None:
            ignore.append(item)
            continue
        #Grab match and add to list
        #And make lowercase for simplicity and to remove duplicates of different case later on
        taxon = match.group(1).lower()
        taxa.append(taxon)

    #Remove duplicates and sort
    unique_taxons = list(set(taxa))
    unique_taxons.sort()

    #Write lists to files
    write_to_file("config.txt",unique_taxons)
    write_to_file("ignore.txt",ignore)


    return unique_taxons


def main():
    
    taxa = get_taxa()
    
if __name__ == "__main__":
    main()
