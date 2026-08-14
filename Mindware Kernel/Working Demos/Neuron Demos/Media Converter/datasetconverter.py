import os
import sys
import time
import numpy as np
import cv2

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("[WARNING] librosa not installed — audio conversion disabled.")

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

# ============ CONFIG ============
IMG_TARGET_SIZE = (64, 64)
AUDIO_SR        = 16000
AUDIO_CHUNK_S   = 1.0
AUDIO_CHUNK_N   = int(AUDIO_SR * AUDIO_CHUNK_S)
N_MELS          = 64
SPEC_HEIGHT     = 64
SPEC_WIDTH      = 64
N_FFT           = 512
HOP_LENGTH      = 250

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
AUDIO_EXTS = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac'}
# ================================


def print_header(title):
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def scan_files(folder, extensions):
    """Recursively collect files with given extensions."""
    found = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f.lower())[1] in extensions:
                found.append(os.path.join(root, f))
    return sorted(found)


def process_image_file(src_path, out_dir, idx):
    """Load, resize, RGB-convert, normalize [0,1], save as .npy"""
    try:
        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            return False, "cv2 failed to decode"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_TARGET_SIZE, interpolation=cv2.INTER_AREA)
        norm = img.astype(np.float32) / 255.0
        out_name = f"frame_{idx:05d}.npy"
        np.save(os.path.join(out_dir, out_name), norm)
        return True, out_name
    except Exception as e:
        return False, str(e)


def audio_to_spec(audio):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=AUDIO_SR, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS, center=True
    )
    if mel.shape[1] < SPEC_WIDTH:
        mel = np.pad(mel, ((0, 0), (0, SPEC_WIDTH - mel.shape[1])), mode='edge')
    mel = mel[:, :SPEC_WIDTH]
    if mel.shape[0] < SPEC_HEIGHT:
        mel = np.pad(mel, ((0, SPEC_HEIGHT - mel.shape[0]), (0, 0)), mode='edge')
    mel = mel[:SPEC_HEIGHT, :]

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db + 80.0) / 80.0
    mel_norm = np.clip(mel_norm, 0.0, 1.0)
    return mel_norm.astype(np.float32)[..., np.newaxis]


def load_audio_any(path):
    """Load audio with librosa or soundfile, resample to AUDIO_SR, mono."""
    try:
        y, sr = librosa.load(path, sr=AUDIO_SR, mono=True)
        return y
    except Exception:
        if HAS_SOUNDFILE:
            y, sr = sf.read(path, dtype='float32')
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            if sr != AUDIO_SR:
                y = librosa.resample(y, orig_sr=sr, target_sr=AUDIO_SR)
            return y
    return None


def process_audio_file(src_path, out_dir, base_idx):
    """Load audio, slice into 1s chunks, convert to mel-spec, save as .npy"""
    try:
        audio = load_audio_any(src_path)
        if audio is None or len(audio) < AUDIO_CHUNK_N // 2:
            return 0, "failed to load or too short"

        n_chunks = len(audio) // AUDIO_CHUNK_N
        saved = 0
        for i in range(n_chunks):
            chunk = audio[i * AUDIO_CHUNK_N:(i + 1) * AUDIO_CHUNK_N]
            spec = audio_to_spec(chunk)
            out_name = f"audio_{base_idx + saved:05d}.npy"
            np.save(os.path.join(out_dir, out_name), spec)
            saved += 1
        return saved, "ok"
    except Exception as e:
        return 0, str(e)


def convert_images():
    print_header("Image Dataset Converter")
    src = input("Source folder with images: ").strip().strip('"')
    if not os.path.isdir(src):
        print(f"[ERROR] Not a directory: {src}")
        return

    out = input("Output folder (default: ./dataset): ").strip().strip('"')
    if not out:
        out = "dataset"
    os.makedirs(out, exist_ok=True)

    files = scan_files(src, IMG_EXTS)
    print(f"Found {len(files)} image(s).")
    if not files:
        return

    print(f"Processing into {out}/ ...")
    ok = 0
    fail = 0
    t0 = time.time()

    for i, fpath in enumerate(files):
        success, msg = process_image_file(fpath, out, i)
        if success:
            ok += 1
            print(f"  [{i+1}/{len(files)}] {msg}")
        else:
            fail += 1
            print(f"  [{i+1}/{len(files)}] FAIL: {os.path.basename(fpath)} — {msg}")

    elapsed = time.time() - t0
    print("-" * 40)
    print(f"Done: {ok} saved, {fail} failed in {elapsed:.2f}s")
    print(f"Output: {os.path.abspath(out)}/")


def convert_audio():
    print_header("Audio Dataset Converter")
    if not HAS_LIBROSA:
        print("[ERROR] librosa is required. Run: pip install librosa soundfile")
        return

    src = input("Source folder with audio: ").strip().strip('"')
    if not os.path.isdir(src):
        print(f"[ERROR] Not a directory: {src}")
        return

    out = input("Output folder (default: ./audio_dataset): ").strip().strip('"')
    if not out:
        out = "audio_dataset"
    os.makedirs(out, exist_ok=True)

    files = scan_files(src, AUDIO_EXTS)
    print(f"Found {len(files)} audio file(s).")
    if not files:
        return

    print(f"Processing into {out}/ ...")
    total_chunks = 0
    fail = 0
    t0 = time.time()
    idx = 0

    for i, fpath in enumerate(files):
        n_saved, msg = process_audio_file(fpath, out, idx)
        if n_saved > 0:
            total_chunks += n_saved
            idx += n_saved
            print(f"  [{i+1}/{len(files)}] {os.path.basename(fpath)} -> {n_saved} chunk(s)")
        else:
            fail += 1
            print(f"  [{i+1}/{len(files)}] FAIL: {os.path.basename(fpath)} — {msg}")

    elapsed = time.time() - t0
    print("-" * 40)
    print(f"Done: {total_chunks} chunk(s) from {len(files)-fail} file(s) in {elapsed:.2f}s")
    print(f"Output: {os.path.abspath(out)}/")


def main():
    print_header("SNN Dataset Preparation Tool")
    print("  Convert raw media into .npy training data for the SNNs.")
    print("-" * 60)
    print("  [1] Convert images  -> dataset/      (64x64x3 RGB .npy)")
    print("  [2] Convert audio   -> audio_dataset/ (64x64x1 mel-spec .npy)")
    print("  [q] Quit")
    print("-" * 60)

    choice = input("Choice: ").strip().lower()

    if choice == '1':
        convert_images()
    elif choice == '2':
        convert_audio()
    elif choice in ('q', 'quit', 'exit'):
        sys.exit(0)
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
