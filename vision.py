import cv2
import numpy as np
import time
import glob
import os
import sys
from svd import svd_hand

def has_meaningful_object(edges, min_edge_pixels=50, min_bounding_area=100):
    """Check if the edge image contains a meaningful object (not just noise or blank)."""
    coords = cv2.findNonZero(edges)
    if coords is None:
        return False
    x, y, w, h = cv2.boundingRect(coords)
    if w * h < min_bounding_area:
        return False
    if len(coords) < min_edge_pixels:
        return False
    return True

def get_aligned_gray(image, crop=True):
    """Extract the aligned 100x100 grayscale image using the same alignment as edges."""
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    if crop:
        box_h = int(h * 0.8)
        box_w = int(box_h * 0.4)
        y1 = h//2 - box_h//2
        x1 = w//2 - box_w//2
        processed = image[y1:y1+box_h, x1:x1+box_w]
    else:
        processed = image
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    coords = cv2.findNonZero(edges)
    if coords is not None:
        x, y, w_c, h_c = cv2.boundingRect(coords)
        if w_c > 0 and h_c > 0:
            cropped = gray[y:y+h_c, x:x+w_c]
            scale = min(100.0 / w_c, 100.0 / h_c)
            new_w, new_h = int(w_c * scale), int(h_c * scale)
            resized = cv2.resize(cropped, (new_w, new_h))
            padded = np.zeros((100, 100), dtype=np.uint8)
            y_off, x_off = (100 - new_h) // 2, (100 - new_w) // 2
            padded[y_off:y_off+new_h, x_off:x_off+new_w] = resized
            gray = padded
        else:
            gray = cv2.resize(gray, (100, 100))
    else:
        gray = cv2.resize(gray, (100, 100))
    return gray

def gray_correlation(ref_image, cap_image, crop=True):
    """Compute Pearson correlation between aligned grayscale images.
    Returns a value between -1 and 1. Higher means more similar."""
    ref_gray = get_aligned_gray(ref_image, crop)
    cap_gray = get_aligned_gray(cap_image, crop)
    if ref_gray is None or cap_gray is None:
        return None
    ref_vec = ref_gray.astype(float).flatten()
    cap_vec = cap_gray.astype(float).flatten()
    ref_mean, cap_mean = np.mean(ref_vec), np.mean(cap_vec)
    ref_std, cap_std = np.std(ref_vec), np.std(cap_vec)
    if ref_std < 1e-10 or cap_std < 1e-10:
        return 0.0
    corr = np.mean((ref_vec - ref_mean) * (cap_vec - cap_mean)) / (ref_std * cap_std)
    return corr

def extract_features(image, k=15, crop=True, require_object=True):
    """ Extracts top k singular values of the image as its features. 
    Returns None if no meaningful object is detected in the frame. """
    if image is None or image.size == 0:
        return None
        
    h, w = image.shape[:2]
    
    if crop:
        # 1. Decupăm doar un dreptunghi din centrul imaginii (pentru a elimina fundalul stânga/dreapta)
        box_h = int(h * 0.8)
        box_w = int(box_h * 0.4) 
        
        y1 = h//2 - box_h//2
        x1 = w//2 - box_w//2
        processed_image = image[y1:y1+box_h, x1:x1+box_w]
    else:
        processed_image = image
        
    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    
    # 2. Aplicăm blur și Canny Edge Detection pentru a păstra doar CONTURUL obiectului
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    
    # --- NOU: Izolăm obiectul și îl centrăm, PĂSTRÂND PROPORȚIILE ---
    # Găsim toate punctele albe (marginile) și facem un chenar perfect în jurul lor
    coords = cv2.findNonZero(edges)
    if coords is not None:
        x, y, w_c, h_c = cv2.boundingRect(coords)
        if w_c > 0 and h_c > 0:
            cropped = edges[y:y+h_c, x:x+w_c]
            # Scalăm proporțional astfel încât latura maximă să fie de 100 (invarianță la distanță)
            scale = min(100.0 / w_c, 100.0 / h_c)
            new_w, new_h = int(w_c * scale), int(h_c * scale)
            resized = cv2.resize(cropped, (new_w, new_h))
            
            # Plasăm forma perfect în centrul unei matrici de 100x100 (invarianță la tremurat)
            padded = np.zeros((100, 100), dtype=np.uint8)
            y_off, x_off = (100 - new_h) // 2, (100 - new_w) // 2
            padded[y_off:y_off+new_h, x_off:x_off+new_w] = resized
            edges = padded
        else:
            edges = cv2.resize(edges, (100, 100))
    else:
        edges = cv2.resize(edges, (100, 100))
    
    # Check if there's a meaningful object before proceeding
    if require_object and not has_meaningful_object(edges):
        return None
        
    normalized = edges.astype(float) / 255.0
    
    # Acum aplicăm SVD. Sigma va surprinde impecabil forma, ignorând tremuratul!
    _, Sigma, _ = svd_hand(normalized, k)
    
    norm = np.linalg.norm(Sigma)
    if norm > 0:
        Sigma = Sigma / norm
        
    return Sigma

def is_toothbrush(captured_image, reference_features, threshold=0.15, captured_features=None):
    if captured_features is None:
        captured_features = extract_features(captured_image)
    if captured_features is None:
        print("⚠️ No meaningful object detected in the frame.")
        return False
    if len(captured_features) != len(reference_features):
        print("\n❌ EROARE: Profilul vechi este incompatibil cu noul SVD (Dimensiuni diferite).")
        print("❌ Te rog reporneste aplicatia si alege 'da' la utilizator nou pentru a face poza!\n")
        return False
    # Folosim Distanța Cosinus (standard pentru feature vectors normalizați)
    similarity = np.dot(captured_features, reference_features)
    distance = 1.0 - similarity
    print(f"Image distance (Cosine) to reference: {distance:.4f} (Threshold: {threshold})")
    return distance < threshold

def validate_against_database(captured_image, dataset_dir="dataset/toothbrush", threshold=0.30):
    """ Validează dacă imaginea capturată seamănă cu vreo periuță din setul de date. """
    if not os.path.exists(dataset_dir):
        print(f"Atenție: Folderul {dataset_dir} nu există. Se sare peste validare.")
        return True
        
    db_images = glob.glob(os.path.join(dataset_dir, '*.*'))
    if not db_images:
        print(f"Atenție: Nu există imagini în {dataset_dir}. Se sare peste validare.")
        return True
        
    captured_features = extract_features(captured_image, crop=True)
    if captured_features is None:
        print("⚠️ Nu a fost detectat niciun obiect pentru validare în baza de date.")
        return False
    min_dist = float('inf')
    
    for db_img_path in db_images:
        db_img = cv2.imread(db_img_path)
        if db_img is None: continue
        # Nu decupăm pozele din dataset deoarece au deja alte proporții
        db_features = extract_features(db_img, crop=False)
        dist = 1.0 - np.dot(captured_features, db_features)
        if dist < min_dist: min_dist = dist
            
    print(f"Cea mai mică distanță (Cosine) față de o periuță din DB: {min_dist:.4f} (Prag: {threshold})")
    return min_dist < threshold

def capture_image():
    # Folosim cv2.CAP_DSHOW pe Windows pentru compatibilitate mai bună
    if sys.platform == 'win32':
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Eroare: Nu am putut accesa camera web. Verifică dacă nu e folosită de altă aplicație!")
        return None
        
    # --- NOU: Perioadă de încălzire pentru a permite camerei să își ajusteze expunerea ---
    print("Incalzire camera...")
    for _ in range(30): # Citim și aruncăm primele 30 de cadre
        cap.read()

    print("Camera a pornit! Fereastra se va deschide acum.")
    
    captured_frame = None
    while True:
        ret, frame = cap.read()
        display_frame = frame.copy()
        if not ret:
            print("Eroare la preluarea cadrului.")
            break
            
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Desenăm pe ecran fix același dreptunghi de ghidaj în care se va face decuparea (Crop)
        box_h, box_w = int(h * 0.8), int(h * 0.4)
        y1, x1 = h//2 - box_h//2, w//2 - box_w//2
        
        # --- NOU: Afișăm LIVE conturul exact așa cum îl "vede" SVD-ul ---
        live_crop = frame[y1:y1+box_h, x1:x1+box_w]
        live_gray = cv2.cvtColor(live_crop, cv2.COLOR_BGR2GRAY)
        live_blur = cv2.GaussianBlur(live_gray, (5, 5), 0)
        live_edges = cv2.Canny(live_blur, 40, 120)
        
        coords = cv2.findNonZero(live_edges)
        if coords is not None:
            cx, cy, cw, ch = cv2.boundingRect(coords)
            cv2.rectangle(display_frame, (x1+cx, y1+cy), (x1+cx+cw, y1+cy+ch), (0, 0, 255), 2)
            
        display_frame[y1:y1+box_h, x1:x1+box_w] = cv2.cvtColor(live_edges, cv2.COLOR_GRAY2BGR)
        
        cv2.rectangle(display_frame, (x1, y1), (x1+box_w, y1+box_h), (0, 255, 0), 2)
        cv2.putText(display_frame, "Chenarul rosu prinde forma! -> SPACE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # --- NOU: Verificăm dacă imaginea este validă (nu este neagră) ---
        # Calculăm media pixelilor; dacă e foarte mică, cadrul e probabil negru.
        if np.mean(frame) < 15:
            cv2.putText(display_frame, "Asteptare semnal camera...", (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Camera Web - Apasa SPACE", display_frame)
            if cv2.waitKey(1) & 0xFF == 27: break # Permitem ieșirea cu ESC
            continue # Sarim peste acest cadru și încercăm următorul

        cv2.imshow("Camera Web - Apasa SPACE", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Tasta SPACE
            captured_frame = frame
            break
        elif key == 27:  # Tasta ESC
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return captured_frame