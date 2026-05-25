# Toothbrush SVD Alarm Clock

A university project in Python that creates an alarm clock which only turns off when you take a picture of your toothbrush.

## Requirements Met
- ✅ Python implementation.
- ✅ Works as an alarm that requires a picture to turn off.
- ✅ Uses a Singular Value Decomposition (SVD) algorithm written by hand (using the Power Iteration method).

## Setup Instructions

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the Common Objects in Bathroom Dataset from Kaggle.
3. Extract the downloaded folders (`toothbrush`, `soap`, etc.) into a folder named `dataset` in this directory.
4. Take a picture of your toothbrush using your webcam and save it in this directory as `reference_toothbrush.jpg`.
5. Run the alarm:
   ```bash
   python alarm.py
   ```
6. When the alarm triggers, follow the terminal instructions to show your toothbrush to your computer's webcam.

## Changes & Improvements

- **Cross-platform support** — removed Windows-only `winsound` and `cv2.CAP_DSHOW`; alarm now works on macOS and Linux using terminal bell / native camera backends.
- **SVD robustness fix** — fixed custom power-iteration SVD to handle zero/empty matrices gracefully, preventing `NaN` and `divide-by-zero` warnings when no object is in the frame.
- **Object presence detection** — `extract_features()` now returns `None` when the frame contains no meaningful edges (blank wall, empty hand), giving clear feedback instead of comparing noise.
- **Grayscale correlation check** — added a second verification layer that compares the aligned grayscale images using Pearson correlation. This catches false positives (phones, spray cans) that pure SVD edge features miss.
- **Negative reference learning** — the system now maintains a `negatives/` folder. If a false positive turns off the alarm, the user can press `n` to save that capture as a negative. Future captures that look more like a saved negative than the reference toothbrush are rejected automatically.
- **Removed user registration prompt** — the alarm now auto-registers on first run if no `reference_toothbrush.jpg` exists, streamlining the flow.