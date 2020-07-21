import os

params = [
        "-i 76 1 6 -y 72 58 73 65 55 95 -x 2 2 0 8 10 -2 -s 0 -l 2 2 1 0 1 1 -f dnn", 
        "-i 74 1 6 -y 55 47 54 56 49 48 -x -8 -6 -4 -2 0 0 -s 0 -l 1 1 1 0 0 0 -f dnn",
        "-i 74 2 8 -y 83 70 75 71 67 65 61 62 -x -3 -3 -12 -1 1 3 1 3 -s 0 -l 2 2 1 0 0 0 2 2 -f dnn",
        "-i 74 2 9 -y 83 70 75 71 67 65 61 62 60 -x -3 -3 -12 -1 1 3 1 3 -4 -s 0 -l 2 2 1 0 0 0 2 2 2 -f dnn",
        "-i 75 1 3 -y 83 80 104 -x -3 -2 -12 -s 0 -l 2 2 0 -f dnn"]
for p in params:
    command = "/home/madina_abdrakhmanova/miniconda3/bin/python align_crop_image.py -p /mnt/sharefolder/Drive/thermal_db/train_data/ "+p
    print(command)
    os.system(command)

