
print("Menginstal library ultralytics dan roboflow...")
%pip install ultralytics roboflow -q
print("Instalasi selesai.")



import os
import shutil
import glob
from roboflow import Roboflow
from kaggle_secrets import UserSecretsClient

print("Mengambil API key dari Kaggle Secrets...")
user_secrets = UserSecretsClient()
ROBOFLOW_API_KEY = user_secrets.get_secret("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=ROBOFLOW_API_KEY)

WORKSPACE_ID = "xxxxx"

HUMAN_PROJECT_ID = "zzzzz" 

LEGO_PROJECT_ID = "yyyyy"    


KAGGLE_WORKING_DIR = "/kaggle/working/"
print(f"Direktori kerja: {KAGGLE_WORKING_DIR}")


print("\n--- Mengunduh Dataset Manusia (Proyek 1) ---")
project_human = rf.workspace(WORKSPACE_ID).project(HUMAN_PROJECT_ID)
dataset_human = project_human.versions()[-1].download("yolov8", location=os.path.join(KAGGLE_WORKING_DIR, "source_human"))

print("\n--- Mengunduh Dataset Lego (Proyek 2) ---")
project_lego = rf.workspace(WORKSPACE_ID).project(LEGO_PROJECT_ID)
dataset_lego = project_lego.versions()[-1].download("yolov8", location=os.path.join(KAGGLE_WORKING_DIR, "source_lego"))


print("\n" + "="*50)
print("MEMULAI PROSES PENGGABUNGAN DATASET")
print("="*50)


MIXED_DATASET_DIR = os.path.join(KAGGLE_WORKING_DIR, "mixed_dataset")

def combine_datasets(human_path, lego_path, output_path):
    
    for folder in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_path, 'images', folder), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'labels', folder), exist_ok=True)

    
    def process_and_copy(source_dir, dest_dir, new_class_id=None):
        for split in ['train', 'valid', 'test']:
            image_files = glob.glob(os.path.join(source_dir, split, 'images', '*.jpg'))
            
            for img_path in image_files:
                base_name = os.path.basename(img_path)
                label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')

                
                shutil.copy(img_path, os.path.join(dest_dir, 'images', split, base_name))

                
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f_in:
                        lines = f_in.readlines()
                    
                    with open(os.path.join(dest_dir, 'labels', split, base_name.replace('.jpg', '.txt')), 'w') as f_out:
                        for line in lines:
                            parts = line.strip().split()
                            if new_class_id is not None:
                                parts[0] = str(new_class_id) # Ganti class ID
                            f_out.write(" ".join(parts) + "\n")

    
    print("Menyalin file dataset manusia (Class ID: 0)...")
    process_and_copy(human_path, output_path)

    
    print("Menyalin file dataset lego dan mengubah Class ID menjadi 1...")
    process_and_copy(lego_path, output_path, new_class_id=1)

    print("Penggabungan dataset selesai.")
    return output_path


combine_datasets(dataset_human.location, dataset_lego.location, MIXED_DATASET_DIR)


yaml_content = f"""
path: {MIXED_DATASET_DIR}
train: images/train
val: images/valid
test: images/test

names:
  0: human
  1: lego_person
"""
mixed_yaml_path = os.path.join(MIXED_DATASET_DIR, "data.yaml")
with open(mixed_yaml_path, 'w') as f:
    f.write(yaml_content)
print(f"File konfigurasi gabungan berhasil dibuat di: {mixed_yaml_path}")



from ultralytics import YOLO

print("\n" + "="*50)
print("MEMULAI TRAINING MODEL CAMPURAN")
print("="*50)

model = YOLO('yolov8n.pt')

results = model.train(
    data=mixed_yaml_path,
    epochs=75,
    imgsz=640,
    project="/kaggle/working/runs/detect",
    name="mixed_human_lego_model"
)
print("--- Training Model Campuran Selesai ---")



print("\n\n" + "="*50)
print("PROSES TRAINING SELESAI!")
print("Hasil training dapat ditemukan di direktori berikut:")
!ls -l /kaggle/working/runs/detect
print("\nBobot model terbaik ('best.pt') ada di dalam sub-folder 'weights'.")
print("="*50)