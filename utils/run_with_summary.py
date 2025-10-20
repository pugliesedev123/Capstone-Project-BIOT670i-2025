# importing subprocess module 
import subprocess
import shlex
import datetime
from datetime import timedelta
import glob
import os

def generate_arguments(argument_dict: dict):
    commands = []

    # Chatgpt help
    for argument, value in argument_dict.items():
        flag = f"--{argument}"
        if isinstance(value, bool):
            if value:
                commands += [flag]
        elif value is None or value == "":
            continue
        elif isinstance(value, (list, tuple)):
            for v in value:
                commands += [f"{flag} {v}"]
        else:
            commands += [f"{flag} {value}"]
        
    return commands

def main():

    # Define your arguments for the run below, they will be exported to a summary file.
    # !!!!Ensure you carefully tune your variables, as runtime is long and failure might not occur until well into the run!!!!

    # see defaults on ./scripts/augment_images.py
    augment_argument_dict = {
        "input-root": "./data/train",                         # declare your target input training folder
        "input-config": "./taxa-config.txt",                  # declare your target taxa-config file (use with include-config-classes-only)
        "val-root": "./data/val/owner-combined",              # declare your target validation root folder
        "aug-root": "./data/augmented/owner-combined",        # declare your target augmentation root folder
        "aug-per-image": 3,                                   # declare a non-zero integer
        "seed": 42,                                           # declare a non-zero integer
        "console-print": True,                                # mark true or false
        "exclude-classes": False,                             # mark true or false
        "include-config-classes-only": False,                 # mark true or false
        "threshold": None,                                    # declare a non-zero integer
        "disable-tf": [],                                     # include transforms you'd like to exclude ['rotate', 'scale', 'zoom', 'horizontalflip', 'verticalflip', 'grayscale', 'equalize', 'sharpen']
        "disable-ca": []                                      # include classes you'd like to exclude ['exogyra_sp', 'pycnodonte_sp', 'isurus_sp', ...]
    }

    # see defaults on ./scripts/train_model.py
    train_argument_dict = {
        "use-augmented": True,                    # mark true or false
        "console-print": False,                   # mark true or false
        "use-pre-train": True,                    # mark true or false
        "seed": 42,                               # declare a non-zero integer
        "batch-size": 16,                         # declare a non-zero integer
        "epochs": 5,                              # declare a non-zero integer
        "input-config": "taxa-config.txt",      # declare your target taxa-config file
        "model-path": None,                       # declare your target model weights path or leave None
        "index-path": None,                       # declare your target embedding index path or leave None
        "model": "resnet18",                      # choose one: resnet18, resnet34, resnet50, vgg16, densenet121
        "threshold": None,                        # declare a non-zero integer or leave None
        "exclude-classes": False,                 # mark true or false
        "include-config-classes-only": False      # mark true or false
    }

    # see defaults on ./scripts/predict_image.py
    predict_argument_dict = {
        "example-dir": "./example",                              # required: path to your example images folder (recurses)
        "console-print": False,                         # mark true or false
        "top-predictions": 3,                           # declare a non-zero integer
        "neighbors": 3,                                 # declare a non-zero integer
        "model-path": "models/fossil_resnet18.pt",      # declare your trained weights file
        "class-names": "models/class_names.json",       # declare your class_names.json path
        "index-path": "models/train_index_resnet18.pt",        # declare your training feature index or leave as placeholder
        "output-dir": "output"                          # declare your target folder for the CSV
    }

    start_time = datetime.datetime.now()

    # subprocess.run(["python", "./utils/taxa_for_config.py"]) #may need to export the taxa used below

    # Run Augmentation Script

    augment_images_commands = generate_arguments(augment_argument_dict)
    augment_argument_summary = f"The following arguments were used to run './scripts/augment_images.py' on {datetime.datetime.now()}:\n{augment_images_commands}"
    flat_aug_args = shlex.split(" ".join(augment_images_commands))

    subprocess.run(["python", "./scripts/augment_images.py", *flat_aug_args], check=True)

    # Run Training Script

    training_commands = generate_arguments(train_argument_dict)
    training_summary = f"The following arguments were used to run './scripts/train_model.py' on {datetime.datetime.now()}:\n{training_commands}"
    flat_train_args = shlex.split(" ".join(training_commands))

    subprocess.run(["python", "./scripts/train_model.py", *flat_train_args], check=True)

    # Run Prediction Script

    prediction_commands = generate_arguments(predict_argument_dict)
    prediction_summary = f"The following arguments were used to run './scripts/predict_image.py' on {datetime.datetime.now()}:\n{prediction_commands}"
    flat_predic_args = shlex.split(" ".join(prediction_commands))

    subprocess.run(["python", "./scripts/predict_image.py", *flat_predic_args], check=True)

    end_time = datetime.datetime.now()
    duration = end_time - start_time

    list_of_files = glob.glob(f"./{predict_argument_dict.get('output-dir', 'output')}/*.csv") # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)

    summary_file = open(latest_file.replace("predictions", "summary").replace(".csv", ".txt"), "w")
    summary_file.write(f"Runtime Results\n------------------------------------------------------------------------------------------\n\nOutput File: {latest_file}\n\n{augment_argument_summary}\n\n{training_summary}\n\n{prediction_summary}")
    summary_file.write(f"\n\nTotal time elapsed: {duration}")

    aug_files = val_files = folders = 0

    for _, dirnames, filenames in os.walk(f"./{augment_argument_dict.get('aug-root')}"):
        aug_files += len(filenames)
        folders += len(dirnames)
    for _, _, filenames in os.walk(f"./{augment_argument_dict.get('val-root')}"):
        val_files += len(filenames)
        folders += len(dirnames)

    summary_file.write(f"\n\nTotal number of classes: {folders}\nTotal number of augmentation files: {aug_files}\nTotal number of validation files: {val_files}")
    summary_file.write(f"\n\nTaxa included: {folders}\nTotal number of augmentation files: {aug_files}\nTotal number of validation files: {val_files}")

    summary_file.close()

if __name__ == "__main__":
    main()