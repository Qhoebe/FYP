import numpy as np
import os

ROOT_DIR = os.path.abspath(os.curdir)
print('root dir:', ROOT_DIR)

with open("output.txt", "w") as f:
    f.write("This is my output.")
    f.write(ROOT_DIR)

