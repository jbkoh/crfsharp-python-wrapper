# Written by Jason Koh
import os

# Adopted from: http://stackoverflow.com/a/8412405
def rolling_window(l, w_size):
    for i in range(len(l)-w_size+1):
        yield [l[i+o] for o in range(w_size)]

def check_and_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
