import os

# calcola il numero di linee dei file che hanno _runs nel nome
directory = "c:\\git\\segmentation\\data\mathtests\\"

for filename in os.listdir(directory):
    if "_runs" in filename:
        filepath = os.path.join(directory, filename)
        with open(filepath, "r") as f:
            num_lines = sum(1 for _ in f)
        print(f"{filename}: {num_lines - 1} lines")
        