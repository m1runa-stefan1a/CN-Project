import time
import cv2
import os
import sys
import glob
import threading
from vision import extract_features, is_toothbrush, capture_image, gray_correlation

# Cross-platform alarm sound setup
if sys.platform == 'win32':
    import winsound
    def _play_beep():
        winsound.Beep(2500, 400)
else:
    # macOS / Linux fallback using terminal bell / system sound
    def _play_beep():
        print('\a', end='', flush=True)

stop_alarm_event = threading.Event()

def beep_thread_worker():
    # Sună continuu și enervant până când este oprit manual (din cod)
    while not stop_alarm_event.is_set():
        _play_beep()
        stop_alarm_event.wait(0.3) # Pauză scurtă între bip-uri

def load_negative_images(negatives_dir='negatives'):
    """Încarcă imaginile negative salvate anterior."""
    negatives = []
    if not os.path.exists(negatives_dir):
        return negatives
    for path in glob.glob(os.path.join(negatives_dir, '*.jpg')):
        img = cv2.imread(path)
        if img is not None:
            negatives.append(img)
    return negatives

def run_alarm():
    print("--- Toothbrush Alarm Project ---")
    ref_path = 'reference_toothbrush.jpg'
    negatives_dir = 'negatives'
    os.makedirs(negatives_dir, exist_ok=True)
    
    # Dacă nu există profil, facem automat înregistrarea
    if not os.path.exists(ref_path):
        print("\n=== INREGISTRARE PERIUTA ===")
        print("Nu am gasit un profil salvat. Te rog sa pozezi periuta ta de dinti in chenarul verde.")
        input("Apasa Enter pentru a porni camera...")
        
        captured_image = capture_image()
        if captured_image is None:
            print("Eroare la captura. Iesire...")
            return
            
        cv2.imwrite(ref_path, captured_image)
        print(f"✅ Periuta ta a fost salvata ca profil ('{ref_path}').\n")

    print("Se incarca profilul...")
    reference_image = cv2.imread(ref_path)
    if reference_image is None:
        print(f"Eroare: Nu am putut incarca '{ref_path}'. Sterge fisierul si ruleaza din nou.")
        return
    reference_features = extract_features(reference_image)
    negative_images = load_negative_images(negatives_dir)
    if negative_images:
        print(f"Loaded {len(negative_images)} negative reference(s).")
    print("Reference setup complete!\n")

    print("Alarm is set. Waiting for trigger...")
    time.sleep(5) # Simulating time until alarm goes off (5 seconds for testing)
    
    print("\n" + "="*40)
    print("⏰ ALARM RINGING! WAKE UP! ⏰")
    print("="*40)
    
    # Pornim sunetul într-un thread separat ca să nu blocheze programul
    stop_alarm_event.clear()
    alarm_thread = threading.Thread(target=beep_thread_worker)
    alarm_thread.daemon = True
    alarm_thread.start()

    alarm_active = True
    while alarm_active:
        input("Press Enter to take a picture of your toothbrush to turn off the alarm...")
        
        captured_image = capture_image()
        if captured_image is not None:
            cv2.imwrite("last_capture.jpg", captured_image)
            features = extract_features(captured_image)
            corr_ref = gray_correlation(reference_image, captured_image)
            if features is None or corr_ref is None:
                print("⚠️ No object detected. Please show your toothbrush clearly in the frame.")
                continue
            
            print(f"Correlation to reference: {corr_ref:.3f}")
            
            # Check against negative references
            neg_reject = False
            best_neg_corr = -1.0
            for neg_img in negative_images:
                corr_neg = gray_correlation(neg_img, captured_image)
                if corr_neg is not None and corr_neg > best_neg_corr:
                    best_neg_corr = corr_neg
                if corr_neg is not None and corr_ref < corr_neg + 0.08:
                    neg_reject = True
                    break
            
            if best_neg_corr >= 0:
                print(f"Best negative correlation: {best_neg_corr:.3f}")
            
            svd_match = is_toothbrush(captured_image, reference_features, captured_features=features)
            corr_match = corr_ref > 0.60
            
            if neg_reject:
                print("❌ This looks like something you've shown before that's NOT your toothbrush.")
            elif svd_match and corr_match:
                print("Toothbrush detected! Alarm stopped. Good morning! ☀️")
                alarm_active = False
                stop_alarm_event.set()
                alarm_thread.join()
                
                # Ask for confirmation to improve future detection
                confirm = input("Was this really your toothbrush? (apasa Enter pentru da, 'n' pentru nu): ").strip().lower()
                if confirm == 'n':
                    neg_path = os.path.join(negatives_dir, f'neg_{int(time.time())}.jpg')
                    cv2.imwrite(neg_path, captured_image)
                    print(f"Saved as negative reference: {neg_path}")
                    print("Next time, similar objects will be rejected automatically.")
            elif not svd_match:
                print("❌ SVD shape mismatch. That doesn't look like your toothbrush.")
            else:
                print("❌ Texture/color mismatch. That doesn't look like your toothbrush.")
        else:
            print("Failed to capture image. Please try again.")
            
if __name__ == "__main__":
    run_alarm()
