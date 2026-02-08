import numpy as np
import requests
import os
import io
import random
import time
from PIL import Image

# ================= SETTINGS =================
API_HOST = "https://k49.mikulab.com" 

# Number of samples to test
SAMPLE_SIZE = 20 

# Official K49 dataset download URLs
K49_URL_IMGS = "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz"
K49_URL_LABELS = "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz"
# ============================================

def download_file(url, filename):
    """Download the dataset if it doesn't exist locally."""
    if not os.path.exists(filename):
        print(f"📥 Downloading {filename}...")
        r = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Download complete.")
    else:
        print(f"📂 {filename} already exists, skipping download.")

def load_k49_data():
    """Load the K49 test set into memory."""
    download_file(K49_URL_IMGS, "k49-test-imgs.npz")
    download_file(K49_URL_LABELS, "k49-test-labels.npz")
    
    print("📦 Loading Numpy arrays...")
    imgs = np.load("k49-test-imgs.npz")['arr_0']
    labels = np.load("k49-test-labels.npz")['arr_0']
    return imgs, labels

def numpy_to_bytes(img_array):
    """Convert a (28,28) numpy array to a PNG byte stream."""
    # KMNIST is natively 0 for black and 255 for white.
    # The API's preprocess.py handles inversion if necessary.
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

def test_sync(imgs, labels):
    """Test the synchronous prediction endpoint."""
    print(f"\n🚀 [Sync Mode] Testing {SAMPLE_SIZE} random samples...")
    correct = 0
    indices = random.sample(range(len(imgs)), SAMPLE_SIZE)
    
    start_time = time.time()
    
    for idx in indices:
        img_bytes = numpy_to_bytes(imgs[idx])
        true_label = labels[idx]
        
        try:
            # Dispatch prediction request
            resp = requests.post(
                f"{API_HOST}/predict",
                files={"file": ("test.png", img_bytes, "image/png")}
            )
            
            if resp.status_code == 200:
                result = resp.json()
                pred_id = result['prediction_id']
                conf = result['confidence']
                char = result['character']
                
                is_correct = (pred_id == true_label)
                if is_correct: correct += 1
                
                mark = "✅" if is_correct else f"❌ (True: {true_label})"
                print(f"   Img {idx:<5} -> Pred: {char} (ID:{pred_id}) Conf:{conf:.2f} {mark}")
            else:
                print(f"   ❌ API Error: {resp.status_code}")
                
        except Exception as e:
            print(f"   ❌ Connection Error: {e}")
            
    total_time = time.time() - start_time
    acc = correct / SAMPLE_SIZE * 100
    print(f"📊 [Sync Result] Accuracy: {acc:.1f}% ({correct}/{SAMPLE_SIZE}) | Avg Latency: {total_time/SAMPLE_SIZE*1000:.0f}ms")

def test_async(imgs, labels):
    """Test the asynchronous batch prediction endpoint."""
    print(f"\n🚀 [Async Mode] Uploading {SAMPLE_SIZE} samples as a batch...")
    indices = random.sample(range(len(imgs)), SAMPLE_SIZE)
    
    # Prepare payload
    files = []
    for i, idx in enumerate(indices):
        img_bytes = numpy_to_bytes(imgs[idx])
        files.append(('files', (f'img_{idx}.png', img_bytes, 'image/png')))
    
    try:
        # 1. Submit batch job
        resp = requests.post(f"{API_HOST}/batch_predict", files=files)
        if resp.status_code not in [200, 202]:
            print(f"❌ Batch upload failed: {resp.text}")
            return
            
        task_ids = resp.json()['task_ids']
        print(f"✅ Batch queued. Waiting for workers to process...")
        
        # 2. Poll for results
        results = {}
        for _ in range(20): # 20-second timeout
            if len(results) == len(task_ids): break
            
            for tid in task_ids:
                if tid in results: continue
                r = requests.get(f"{API_HOST}/tasks/{tid}").json()
                if r['status'] == 'completed':
                    results[tid] = r['result']
                elif r['status'] == 'failed':
                    results[tid] = None
            time.sleep(1)
            
        # 3. Compile metrics
        correct = 0
        print("\n📝 Batch Results:")
        for i, tid in enumerate(task_ids):
            true_idx = indices[i]
            true_label = labels[true_idx]
            res = results.get(tid)
            
            if res:
                pred_id = res['prediction_id']
                is_correct = (pred_id == true_label)
                if is_correct: correct += 1
                mark = "✅" if is_correct else f"❌ (True: {true_label})"
                print(f"   Task {tid[-4:]} -> Pred: {res['character']} (Conf:{res['confidence']:.2f}) {mark}")
            else:
                print(f"   Task {tid[-4:]} -> Failed or Timeout")

        acc = correct / SAMPLE_SIZE * 100
        print(f"📊 [Async Result] Accuracy: {acc:.1f}% ({correct}/{SAMPLE_SIZE})")

    except Exception as e:
        print(f"❌ Async Error: {e}")

if __name__ == "__main__":
    # Load and prep dataset
    images, labels = load_k49_data()
    print(f"📂 Dataset Loaded. Shape: {images.shape}")
    
    # Run test suite
    test_sync(images, labels)
    test_async(images, labels)
