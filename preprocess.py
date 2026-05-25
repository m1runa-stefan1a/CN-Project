import cv2
import numpy as np
import os
import glob

def preprocess_image(image_path, size=(64, 64)):
    """
    Încarcă, transformă în grayscale, redimensionează și vectorizează o imagine.
    """
    # 1. Încărcare imagine
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # 2. Scala de gri (Grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Redimensionare (ex: 64x64 pentru a limita dimensiunea matricii)
    resized = cv2.resize(gray, size)
    
    # 4. Vectorizare (aplatizare într-un vector coloană de dimensiune M)
    # Pentru 64x64, M = 4096
    vector = resized.flatten().astype(float)
    
    # Normalizare între 0 și 1 pentru stabilitate numerică la calculul SVD
    vector = vector / 255.0
    
    return vector

def build_dataset_matrix(dataset_path, positive_class='toothbrush', size=(64, 64)):
    """
    Construiește matricea A (M x N) din imaginile aflate în dataset_path.
    Folderele care nu sunt 'toothbrush' vor fi considerate 'negative'.
    """
    vectors = []
    labels = []
    
    for class_dir_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_dir_name)
        if not os.path.isdir(class_dir):
            continue
            
        label = positive_class if class_dir_name.lower() == positive_class.lower() else 'negative'

        # Extrage toate imaginile din folderul clasei (ex: .jpg, .png)
        for img_file in glob.glob(os.path.join(class_dir, '*.*')):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                vec = preprocess_image(img_file, size)
                if vec is not None:
                    vectors.append(vec)
                    labels.append(label)
    
    if not vectors:
        raise ValueError("Nu au fost găsite imagini valide pentru a construi setul de date. Adaugă imagini în foldere.")
        
    # 5. Crearea matricii mari de date A
    # Așezăm vectorii coloană unul lângă altul => Matrice M x N
    A = np.column_stack(vectors)
    
    return A, labels

if __name__ == "__main__":
    dataset_dir = "dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    
    print(f"Asigură-te că ai descărcat dataset-ul de pe Kaggle și ai extras folderele în '{dataset_dir}'.")
    print("Folderul 'toothbrush' va fi folosit ca țintă, iar restul obiectelor (săpun, etc.) ca exemple negative.")