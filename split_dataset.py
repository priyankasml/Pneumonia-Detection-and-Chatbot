import os
import shutil

# 🔥 YOUR REAL DATASET PATH
original_dataset = r"D:\archive (14)\chest_xray\chest_xray"

# 🔥 Where new 3-class dataset will be created
new_dataset = r"C:\Users\User\PycharmProjects\pythonProject4\dataset"

for split in ["train", "val", "test"]:

    normal_path = os.path.join(original_dataset, split, "NORMAL")
    pneumonia_path = os.path.join(original_dataset, split, "PNEUMONIA")

    print(f"Processing: {split}")

    if not os.path.exists(normal_path):
        print(f"❌ Path not found: {normal_path}")
        continue

    os.makedirs(os.path.join(new_dataset, split, "NORMAL"), exist_ok=True)
    os.makedirs(os.path.join(new_dataset, split, "BACTERIAL"), exist_ok=True)
    os.makedirs(os.path.join(new_dataset, split, "VIRAL"), exist_ok=True)

    # Copy NORMAL images
    for file in os.listdir(normal_path):
        shutil.copy(
            os.path.join(normal_path, file),
            os.path.join(new_dataset, split, "NORMAL", file)
        )

    # Split Pneumonia into BACTERIAL & VIRAL
    for file in os.listdir(pneumonia_path):
        if "bacteria" in file.lower():
            shutil.copy(
                os.path.join(pneumonia_path, file),
                os.path.join(new_dataset, split, "BACTERIAL", file)
            )
        elif "virus" in file.lower():
            shutil.copy(
                os.path.join(pneumonia_path, file),
                os.path.join(new_dataset, split, "VIRAL", file)
            )

print("\n✅ Dataset successfully split into 3 classes!")