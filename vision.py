import cv2
import numpy as np
import os
import sys
from svd import svd_hand

def has_meaningful_object(edges, min_edge_pixels=50, min_bounding_area=100):
    coords = cv2.findNonZero(edges)
    if coords is None:
        return False
    x, y, w, h = cv2.boundingRect(coords)
    if w * h < min_bounding_area or len(coords) < min_edge_pixels:
        return False
    return True

def get_aligned_gray(image, crop=True):
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    if crop:
        box_h = int(h * 0.8)
        box_w = int(box_h * 0.4)
        y1, x1 = h//2 - box_h//2, w//2 - box_w//2
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
            resized = cv2.resize(cropped, (int(w_c * scale), int(h_c * scale)))
            padded = np.zeros((100, 100), dtype=np.uint8)
            y_off, x_off = (100 - resized.shape[0]) // 2, (100 - resized.shape[1]) // 2
            padded[y_off:y_off+resized.shape[0], x_off:x_off+resized.shape[1]] = resized
            return padded
    return cv2.resize(gray, (100, 100))

def gray_correlation(ref_image, cap_image, crop=True):
    ref_gray = get_aligned_gray(ref_image, crop)
    cap_gray = get_aligned_gray(cap_image, crop)
    if ref_gray is None or cap_gray is None:
        return None
    ref_vec, cap_vec = ref_gray.astype(float).flatten(), cap_gray.astype(float).flatten()
    ref_std, cap_std = np.std(ref_vec), np.std(cap_vec)
    if ref_std < 1e-10 or cap_std < 1e-10:
        return 0.0
    return np.mean((ref_vec - np.mean(ref_vec)) * (cap_vec - np.mean(cap_vec))) / (ref_std * cap_std)

def extract_features(image, k=15, crop=True, require_object=True):
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    if crop:
        box_h = int(h * 0.8)
        box_w = int(box_h * 0.4)
        y1, x1 = h//2 - box_h//2, w//2 - box_w//2
        processed_image = image[y1:y1+box_h, x1:x1+box_w]
    else:
        processed_image = image
        
    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    
    coords = cv2.findNonZero(edges)
    if coords is not None:
        x, y, w_c, h_c = cv2.boundingRect(coords)
        if w_c > 0 and h_c > 0:
            cropped = edges[y:y+h_c, x:x+w_c]
            scale = min(100.0 / w_c, 100.0 / h_c)
            resized = cv2.resize(cropped, (int(w_c * scale), int(h_c * scale)))
            padded = np.zeros((100, 100), dtype=np.uint8)
            y_off, x_off = (100 - resized.shape[0]) // 2, (100 - resized.shape[1]) // 2
            padded[y_off:y_off+resized.shape[0], x_off:x_off+resized.shape[1]] = resized
            edges = padded
            
    if require_object and not has_meaningful_object(edges):
        return None
        
    normalized = edges.astype(float) / 255.0
    _, Sigma, _ = svd_hand(normalized, k)
    
    norm = np.linalg.norm(Sigma)
    return Sigma / norm if norm > 0 else Sigma

def is_toothbrush(captured_image, reference_features, threshold=0.15, captured_features=None):
    if captured_features is None:
        captured_features = extract_features(captured_image)
    if captured_features is None:
        return False
    similarity = np.dot(captured_features, reference_features)
    distance = 1.0 - similarity
    print(f"Distanța SVD față de profilul propriu: {distance:.4f} (Prag acceptat: < {threshold})")
    return distance < threshold

def capture_image():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform == 'win32' else cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Eroare: Nu am putut accesa camera web.")
        return None
        
    for _ in range(30): 
        cap.read() # Warm-up cameră

    captured_frame = None
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        box_h, box_w = int(h * 0.8), int(h * 0.4)
        y1, x1 = h//2 - box_h//2, w//2 - box_w//2
        
        live_crop = frame[y1:y1+box_h, x1:x1+box_w]
        live_edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(live_crop, cv2.COLOR_BGR2GRAY), (5, 5), 0), 40, 120)
        
        coords = cv2.findNonZero(live_edges)
        if coords is not None:
            cx, cy, cw, ch = cv2.boundingRect(coords)
            cv2.rectangle(display_frame, (x1+cx, y1+cy), (x1+cx+cw, y1+cy+ch), (0, 0, 255), 2)
            
        display_frame[y1:y1+box_h, x1:x1+box_w] = cv2.cvtColor(live_edges, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(display_frame, (x1, y1), (x1+box_w, y1+box_h), (0, 255, 0), 2)
        cv2.putText(display_frame, "Aliniaza periuta si apasa SPACE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if np.mean(frame) < 15:
            continue

        cv2.imshow("Camera Web", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            captured_frame = frame
            break
        elif key == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return captured_frame