import time
import cv2
import os
import sys
import glob
import threading
from vision import extract_features, is_toothbrush, capture_image, gray_correlation

if sys.platform == 'win32':
    import winsound
    def _play_beep(): winsound.Beep(2500, 400)
else:
    def _play_beep(): print('\a', end='', flush=True)

stop_alarm_event = threading.Event()

def beep_thread_worker():
    while not stop_alarm_event.is_set():
        _play_beep()
        stop_alarm_event.wait(0.3)

def load_negative_images(negatives_dir='negatives'):
    negatives = []
    if not os.path.exists(negatives_dir):
        return negatives
    for path in glob.glob(os.path.join(negatives_dir, '*.jpg')):
        img = cv2.imread(path)
        if img is not None: negatives.append(img)
    return negatives

def run_alarm():
    from datetime import datetime  
    print("--- Toothbrush Alarm Project ---")
    ref_path = 'reference_toothbrush.jpg'
    negatives_dir = 'negatives'
    os.makedirs(negatives_dir, exist_ok=True)
    
    if not os.path.exists(ref_path):
        print("\n=== INREGISTRARE PERIUTA ===")
        print("Nu am gasit un profil salvat. Te rog sa pozezi periuta ta.")
        input("Apasa Enter pentru a porni camera...")
        captured_image = capture_image()
        if captured_image is None:
            print("Eroare la captura. Iesire...")
            return
        cv2.imwrite(ref_path, captured_image)
        print(f"✅ Profil salvat ca '{ref_path}'.\n")

    print("Se incarca profilul local...")
    reference_image = cv2.imread(ref_path)
    if reference_image is None:
        print("Eroare la incarcarea profilului.")
        return
        
    reference_features = extract_features(reference_image)
    negative_images = load_negative_images(negatives_dir)
    
    while True:
        ora_input = input("Introdu ora la care să sune alarma (format HH:MM, de ex. 07:30): ").strip()
        try:
            tinta_ora, tinta_minut = map(int, ora_input.split(':'))
            if 0 <= tinta_ora < 24 and 0 <= tinta_minut < 60:
                break
            print("❌ Ora trebuie să fie între 00-23, iar minutele între 00-59.")
        except ValueError:
            print("❌ Format invalid! Te rog folosește formatul exact HH:MM.")

    print(f"Sistem pregătit. Alarma va suna la {tinta_ora:02d}:{tinta_minut:02d}. Se așteaptă...")
    
    while True:
        acum = datetime.now()
        if acum.hour == tinta_ora and acum.minute == tinta_minut:
            break
        time.sleep(1) 
     
    print("\n⏰ ALARMA SUNA! TREZESTE-TE! ⏰\n")
    stop_alarm_event.clear()
    alarm_thread = threading.Thread(target=beep_thread_worker, daemon=True)
    alarm_thread.start()

    alarm_active = True
    while alarm_active:
        input("Apasa Enter pentru a face poza periutei tale si a opri alarma...")
        captured_image = capture_image()
        
        if captured_image is not None:
            cv2.imwrite("last_capture.jpg", captured_image)
            features = extract_features(captured_image)
            corr_ref = gray_correlation(reference_image, captured_image)
            
            if features is None or corr_ref is None:
                print("⚠️ Obiect nedetectat clar. Incearca din nou.")
                continue
            
            print(f"Corelatie textura cu profilul: {corr_ref:.3f}")
            
            neg_reject = False
            for neg_img in negative_images:
                corr_neg = gray_correlation(neg_img, captured_image)
                if corr_neg is not None and corr_ref < corr_neg + 0.08:
                    neg_reject = True
                    break
            
            svd_match = is_toothbrush(captured_image, reference_features, captured_features=features)
            corr_match = corr_ref > 0.60
            
            if neg_reject:
                print("❌ Obiectul seamana cu ceva ce ai marcat in trecut ca fiind GRESIT.")
            elif svd_match and corr_match:
                print("✅ Periuta detectata cu succes! Alarma oprita. Bună dimineața! ☀️")
                alarm_active = False
                stop_alarm_event.set()
                alarm_thread.join()
                
                confirm = input("A fost intr-adevar periuta ta? (Enter pentru DA, 'n' pentru NU): ").strip().lower()
                if confirm == 'n':
                    neg_path = os.path.join(negatives_dir, f'neg_{int(time.time())}.jpg')
                    cv2.imwrite(neg_path, captured_image)
                    print(f"Salvat ca exemplu negativ: {neg_path}")
            elif not svd_match:
                print("❌ Potrivire geometrica (SVD) esuata. Nu seamana cu periuta ta.")
            else:
                print("❌ Textura sau culoarea nu corespund.")
        else:
            print("Eroare la captura foto.")
if __name__ == "__main__":
    run_alarm()