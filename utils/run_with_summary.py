# importing subprocess module
import subprocess
import shlex
import datetime
import glob
import os
import psutil
import platform

# For subprocess(), ensure "python" is updated to your system's Python alias
# Be mindful of \\ versus / depending on your system's file structure

def generate_arguments(argument_dict: dict):
    commands = []

    # Generate a list of commands for shlex strings.
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

# Convert the "shlex strings" from generate_arguments into a real argv list.
# This avoids Windows path weirdness (backslashes/spaces) and prevents output-dir from getting mangled.
def build_argv(commands: list[str]) -> list[str]:
    argv = []
    for c in commands:
        # If it's a combined "flag value" string, split once.
        # Example: "--output-dir output\\run_..." -> ["--output-dir", "output\\run_..."]
        if " " in c:
            flag, value = c.split(" ", 1)
            argv.extend([flag, value])
        else:
            argv.append(c)
    return argv

# Calculate disk size.
def get_size(num_bytes, suffix="B"):
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if num_bytes < factor:
            return f"{num_bytes:.2f}{unit}{suffix}"
        num_bytes /= factor
    return f"{num_bytes:.2f}P{suffix}"

def main():

    # Define your arguments for the run below, they will be exported to a summary file.
    # !!!!Ensure you carefully tune your variables, as runtime is long and failure might not occur until well into the run!!!!

    # Ensure the base output/ directory exists.
    base_output_dir = "output"
    os.makedirs(base_output_dir, exist_ok=True)

    # Create a unique output dir for THIS run, and thread it through all scripts.
    # Example: output/run_2026-01-23_22-14-03
    run_name = "[INSERT_RUN_NAME_HERE]"
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_output_dir = os.path.join(base_output_dir, f"run_{run_name}_{run_ts}")
    os.makedirs(run_output_dir, exist_ok=True)

    # see defaults on ./scripts/augment_images.py
    augment_argument_dict = {
        "input-root": "./data/train",                         # declare your target input training folder
        "input-config": "./taxa-config.txt",                  # declare your target taxa-config file (use with include-config-classes-only)
        "val-root": "./data/val/owner-combined",              # declare your target validation root folder
        "aug-root": "./data/augmented/owner-combined",        # declare your target augmentation root folder
        "aug-per-image": 7,                                   # declare a non-zero integer
        "val-frac": 0.3,                                      # declare a non-zero float
        "seed": 42,                                           # declare a non-zero integer
        "console-print": True,                                # mark true or false
        "exclude-classes": False,                             # mark true or false
        "include-config-classes-only": True,                  # mark true or false
        "threshold": 50,                                      # declare a non-zero integer
        "disable-tf": ["equalize"],                           # include transforms you'd like to exclude ['rotate', 'scale', 'zoom', 'horizontalflip', 'verticalflip', 'grayscale', 'equalize', 'sharpen']
        "disable-ca": []                                      # include classes you'd like to exclude ['exogyra_sp', 'pycnodonte_sp', 'isurus_sp', ...]
    }

    # see defaults on ./scripts/train_model.py
    train_argument_dict = {
        "use-augmented": True,                    # mark true or false
        "console-print": True,                    # mark true or false
        "use-pre-train": True,                    # mark true or false
        "seed": 42,                               # declare a non-zero integer
        "batch-size": 8,                          # declare a non-zero integer
        "epochs": 15,                             # declare a non-zero integer
        "disable-early-stopping": True,           # mark true or false (OFF SWITCH)
        "patience": 3,                            # stop after N epochs with no appreciable improvement
        "min-delta": 0.001,                       # minimum change required to count as an improvement
        "monitor": "val_loss",                    # choose one: val_loss, val_acc
        "output-dir": run_output_dir,             # folder where logs will be saved (PER-RUN)
        "input-config": "taxa-config.txt",        # declare your target taxa-config file
        "model-path": None,                       # declare your target model weights path or leave None
        "model": "densenet121",                   # choose one: resnet18, resnet34, resnet50, vgg16, densenet121
        "threshold": None,                        # declare a non-zero integer or leave None
        "exclude-classes": False,                 # mark true or false
        "include-config-classes-only": False      # mark true or false
    }

    # see defaults on ./scripts/predict_image.py
    predict_argument_dict = {
        "example-dir": "./tests",                     # required: path to your example images folder (recurses)
        "console-print": False,                         # mark true or false
        "top-predictions": 3,                           # declare a non-zero integer
        "neighbors": 3,                                 # declare a non-zero integer
        "model-path": "models/fossil_resnet18.pt",      # declare your trained weights file
        "class-names": "models/class_names.json",       # declare your class_names.json path
        "output-dir": "output"                          # declare your target folder for the CSV
    }

    # Start run timer for run duration.
    start_time = datetime.datetime.now()

    # subprocess.run(["python", "./utils/taxa_for_config.py"]) # comment out this line if you are using a customized taxa list for the run

    # Run Augmentation Script
    augment_images_commands = generate_arguments(augment_argument_dict)
    augment_argument_summary = (
        f"The following arguments were used to run './scripts/augment_images.py' on {datetime.datetime.now()}:\n"
        f"{augment_images_commands}"
    )

    flat_aug_args = build_argv(augment_images_commands)
    subprocess.run(["python", "./scripts/augment_images.py", *flat_aug_args], check=True)

    # Run Training Script
    training_commands = generate_arguments(train_argument_dict)
    training_summary = (
        f"The following arguments were used to run './scripts/train_model.py' on {datetime.datetime.now()}:\n"
        f"{training_commands}"
    )

    flat_train_args = build_argv(training_commands)
    subprocess.run(["python", "./scripts/train_model.py", *flat_train_args], check=True)

    # Run Prediction Script
    prediction_commands = generate_arguments(predict_argument_dict)
    prediction_summary = (
        f"The following arguments were used to run './scripts/predict_image.py' on {datetime.datetime.now()}:\n"
        f"{prediction_commands}"
    )

    flat_predic_args = build_argv(prediction_commands)
    subprocess.run(["python", "./scripts/predict_image.py", *flat_predic_args], check=True)

    # End timer and calculated run duration.
    end_time = datetime.datetime.now()
    duration = end_time - start_time

    # Find latest prediction CSV inside THIS run's output directory.
    # (recursive=True makes this resilient if you later nest outputs)
    list_of_files = glob.glob(os.path.join(run_output_dir, "**", "*.csv"), recursive=True)
    if not list_of_files:
        raise RuntimeError(f"[ERROR] No CSV outputs were found in: {run_output_dir}")
    latest_file = max(list_of_files, key=os.path.getctime)

    # Write run summary information including arguments, output file(s), and time elapsed.
    summary_path = latest_file.replace("predictions", "summary").replace(".csv", ".txt")
    summary_file = open(summary_path, "w", encoding="utf-8")
    summary_file.write(
        f"=" * 40 + " Prediction Summary: " + "=" * 40 +
        f"\nOutput File: {latest_file}\n\n{augment_argument_summary}\n\n{training_summary}\n\n{prediction_summary}"
    )
    summary_file.write(f"\n\nTotal time elapsed: {duration}\n\n")

    # Write System Information,
    summary_file.write("=" * 40 + " System Information " + "=" * 40 + "\n")
    uname = platform.uname()
    summary_file.write(f"System: {uname.system}\n")
    summary_file.write(f"Node Name: {uname.node}\n")
    summary_file.write(f"Release: {uname.release}\n")
    summary_file.write(f"Version: {uname.version}\n")
    summary_file.write(f"Machine: {uname.machine}\n")
    summary_file.write(f"Processor: {uname.processor}\n\n")

    # CPU Information,
    summary_file.write("=" * 40 + " CPU Info " + "=" * 40 + "\n")
    summary_file.write(f"Physical cores: {psutil.cpu_count(logical=False)}\n")
    summary_file.write(f"Total cores: {psutil.cpu_count(logical=True)}\n")

    cpufreq = psutil.cpu_freq()
    if cpufreq is not None:
        summary_file.write(f"Max Frequency: {cpufreq.max:.2f}Mhz\n")
        summary_file.write(f"Min Frequency: {cpufreq.min:.2f}Mhz\n")
        summary_file.write(f"Current Frequency: {cpufreq.current:.2f}Mhz\n")
    else:
        summary_file.write("CPU frequency: unavailable\n")

    summary_file.write("CPU Usage Per Core:\n")
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        summary_file.write(f"Core {i}: {percentage}%\n")
    summary_file.write(f"Total CPU Usage: {psutil.cpu_percent()}%\n\n")

    # Memory Information,
    summary_file.write("=" * 40 + " Memory Information " + "=" * 40 + "\n")
    svmem = psutil.virtual_memory()
    summary_file.write(f"Total: {get_size(svmem.total)}\n")
    summary_file.write(f"Available: {get_size(svmem.available)}\n")
    summary_file.write(f"Used: {get_size(svmem.used)}\n")
    summary_file.write(f"Percentage: {svmem.percent}%\n\n")

    summary_file.write("=" * 20 + " SWAP " + "=" * 20 + "\n")
    swap = psutil.swap_memory()
    summary_file.write(f"Total: {get_size(swap.total)}\n")
    summary_file.write(f"Free: {get_size(swap.free)}\n")
    summary_file.write(f"Used: {get_size(swap.used)}\n")
    summary_file.write(f"Percentage: {swap.percent}%\n\n")

    # Disk Information,
    summary_file.write("=" * 40 + " Disk Information " + "=" * 40 + "\n")
    summary_file.write("Partitions and Usage:\n")
    for partition in psutil.disk_partitions():
        summary_file.write(f"=== Device: {partition.device} ===\n")
        summary_file.write(f"  Mountpoint: {partition.mountpoint}\n")
        summary_file.write(f"  File system type: {partition.fstype}\n")
        try:
            usage = psutil.disk_usage(partition.mountpoint)
        except PermissionError:
            continue
        summary_file.write(f"  Total Size: {get_size(usage.total)}\n")
        summary_file.write(f"  Used: {get_size(usage.used)}\n")
        summary_file.write(f"  Free: {get_size(usage.free)}\n")
        summary_file.write(f"  Percentage: {usage.percent}%\n")

    # and Read/Write Information.
    dio = psutil.disk_io_counters()
    if dio is not None:
        summary_file.write(f"Total read: {get_size(dio.read_bytes)}\n")
        summary_file.write(f"Total write: {get_size(dio.write_bytes)}\n\n")

    # Taxa information.
    summary_file.write(f"=" * 40 + " Taxa Information " + "=" * 40 + "\n")
    aug_files = val_files = folders = prev_count_aug = prev_count_val = 0
    taxa_list = {}

    for root, dirname, filenames in os.walk(f"./{augment_argument_dict.get('aug-root')}"):
        aug_files += len(filenames)
        folders += len(dirname)
        if root != "././data/augmented/owner-combined":
            taxa_list[root.replace("././data/augmented/owner-combined\\taxon-", "")] = [len(filenames)]
    for root, dirname, filenames in os.walk(f"./{augment_argument_dict.get('val-root')}"):
        val_files += len(filenames)
        if root != "././data/val/owner-combined":
            taxa_list[root.replace("././data/val/owner-combined\\taxon-", "")].insert(1, len(filenames))

    # Write taxa counts and number.
    summary_file.write(f"Total number of classes: {folders}\nTotal number of training files: {aug_files}\nTotal number of validation files: {val_files}")
    summary_file.write("\n\nTaxa included (tab-delimited):\n\nTaxa\taug_file_count\tval_file_count\n")
    for key, value in taxa_list.items():
        summary_file.write(f"{key}\t{value[0]}\t{value[1]}\n")

    summary_file.close()
    print(f"[INFO] Run outputs saved to: {run_output_dir}")
    print(f"[INFO] Run summary saved to: {summary_path}")

if __name__ == "__main__":
    main()