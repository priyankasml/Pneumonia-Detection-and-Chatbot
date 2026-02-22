import os

base = r"C:\Users\User\OneDrive\Desktop"

for root, dirs, files in os.walk(base):
    if "train" in dirs:
        print("Found train folder at:")
        print(root)
        break