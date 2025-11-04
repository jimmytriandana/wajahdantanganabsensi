import os
import cv2
import time
import threading
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, Response, request, redirect, url_for, flash
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import pickle

mp_hands = mp.solutions.hands

APP_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(APP_DIR, "models")
FACE_CLS_PATH = os.path.join(MODELS_DIR, "face_classifier.h5")
FACE_LE_PATH  = os.path.join(MODELS_DIR, "face_label_encoder.pkl")

HAND_CLS_PATH = os.path.join(MODELS_DIR, "hand_gesture_model.h5")
HAND_LE_PATH  = os.path.join(MODELS_DIR, "hand_label_encoder.pkl")

DATASET_FACES_DIR = os.path.join(APP_DIR, "datasets", "faces")
DATASET_HANDS_DIR = os.path.join(APP_DIR, "datasets", "hands")

os.makedirs(DATASET_FACES_DIR, exist_ok=True)
os.makedirs(DATASET_HANDS_DIR, exist_ok=True)

try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

def clamp_box(x1, y1, x2, y2, W, H):
    return max(0,x1), max(0,y1), min(W-1,x2), min(H-1,y2)

def put_shadow_text(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.6, color=(255,255,255), thick=2):
    x, y = org
    cv2.putText(img, text, (x+1, y+1), font, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)

def draw_error_frame(msg_lines, w=960, h=540):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    y = 40
    for line in msg_lines:
        put_shadow_text(img, line, (20, y), scale=0.8, color=(0,200,255))
        y += 28
    return img

def file_status():
    return {
        "face_classifier": {"path": FACE_CLS_PATH, "exists": os.path.isfile(FACE_CLS_PATH)},
        "face_label_encoder": {"path": FACE_LE_PATH, "exists": os.path.isfile(FACE_LE_PATH)},
        "hand_classifier": {"path": HAND_CLS_PATH, "exists": os.path.isfile(HAND_CLS_PATH)},
        "hand_label_encoder": {"path": HAND_LE_PATH, "exists": os.path.isfile(HAND_LE_PATH)},
    }

class FacePredictor:
    def __init__(self, threshold=0.6, device=None):
        self.threshold = float(threshold)
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.mtcnn = MTCNN(image_size=160, margin=0, keep_all=True,
                           thresholds=[0.6, 0.7, 0.7], factor=0.709,
                           post_process=True, device=self.device)
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.clf = None
        self.le = None
        self.error = None

        try:
            if not os.path.isfile(FACE_CLS_PATH):
                raise FileNotFoundError(f"Face classifier not found at: {FACE_CLS_PATH}")
            if not os.path.isfile(FACE_LE_PATH):
                raise FileNotFoundError(f"Face LabelEncoder not found at: {FACE_LE_PATH}")
            self.clf = load_model(FACE_CLS_PATH, compile=False)
            with open(FACE_LE_PATH, "rb") as f:
                self.le = pickle.load(f)
        except Exception as e:
            self.error = str(e)

    @torch.no_grad()
    def predict(self, frame_bgr):
        if self.error or self.clf is None or self.le is None:
            return [], self.error or "Face model is not loaded"
        H, W = frame_bgr.shape[:2]
        img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        faces = self.mtcnn(img_pil)
        boxes, _ = self.mtcnn.detect(img_pil)
        out = []
        if faces is None or boxes is None:
            return out, None

        if isinstance(faces, torch.Tensor):
            faces = faces.to(self.device)
        else:
            faces = torch.stack(faces).to(self.device)

        embs = self.facenet(faces).cpu().numpy()
        probs = self.clf.predict(embs, verbose=0)
        best_idx = np.argmax(probs, axis=1)
        best_prob = probs[np.arange(probs.shape[0]), best_idx]

        for i, box in enumerate(boxes):
            if box is None: continue
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)
            name = self.le.inverse_transform([best_idx[i]])[0]
            p = float(best_prob[i])
            if p < self.threshold:
                name = "Unknown"
            out.append({"box": (x1, y1, x2, y2), "name": name, "prob": p})
        return out, None

    @torch.no_grad()
    def aligned_face(self, frame_bgr):
        img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        face = self.mtcnn(img_pil)
        if face is None:
            return None
        if isinstance(face, torch.Tensor):
            arr = (face[0] if face.ndim==4 else face).permute(1,2,0).cpu().numpy()*255.0
        else:
            face = face[0] if isinstance(face, (list, tuple)) else face
            arr = face.permute(1,2,0).cpu().numpy()*255.0
        return arr.astype(np.uint8)

class HandPredictor:
    def __init__(self, threshold=0.6):
        self.threshold = float(threshold)
        self.model = None
        self.le = None
        self.error = None
        try:
            if not os.path.isfile(HAND_CLS_PATH):
                raise FileNotFoundError(f"Hand classifier not found at: {HAND_CLS_PATH}")
            if not os.path.isfile(HAND_LE_PATH):
                raise FileNotFoundError(f"Hand LabelEncoder not found at: {HAND_LE_PATH}")
            self.model = load_model(HAND_CLS_PATH, compile=False)
            with open(HAND_LE_PATH, "rb") as f:
                self.le = pickle.load(f)
        except Exception as e:
            self.error = str(e)

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def predict(self, frame_bgr):
        if self.error or self.model is None or self.le is None:
            return [], self.error or "Hand model is not loaded"
        H, W = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out = self.hands.process(frame_rgb)
        results = []
        if not out.multi_hand_landmarks:
            return results, None

        for hlm in out.multi_hand_landmarks:
            xs = [int(lm.x * W) for lm in hlm.landmark]
            ys = [int(lm.y * H) for lm in hlm.landmark]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            mx = int(0.2 * (x2 - x1 + 1))
            my = int(0.2 * (y2 - y1 + 1))
            x1, y1, x2, y2 = clamp_box(x1 - mx, y1 - my, x2 + mx, y2 + my, W, H)
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            img = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
            pred = self.model.predict(np.expand_dims(img, 0), verbose=0)[0]
            j = int(np.argmax(pred)); p = float(pred[j])
            name = self.le.inverse_transform([j])[0] if p >= self.threshold else "Unknown"
            results.append({"box": (x1, y1, x2, y2), "name": name, "prob": p})
        return results, None

    def crop_first_hand(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out = self.hands.process(frame_rgb)
        if not out.multi_hand_landmarks:
            return None
        hlm = out.multi_hand_landmarks[0]
        xs = [int(lm.x * W) for lm in hlm.landmark]
        ys = [int(lm.y * H) for lm in hlm.landmark]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        mx = int(0.2 * (x2 - x1 + 1))
        my = int(0.2 * (y2 - y1 + 1))
        x1, y1, x2, y2 = clamp_box(x1 - mx, y1 - my, x2 + mx, y2 + my, W, H)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        img = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
        return img 

class VideoProcessor:
    def __init__(self, cam_index=0):
        self.lock = threading.Lock()
        self.face = FacePredictor(threshold=0.6)
        self.hand = HandPredictor(threshold=0.6)

        self.cap = cv2.VideoCapture(cam_index)
        self.cam_error = None
        if not self.cap.isOpened():
            self.cam_error = "Cannot open camera (cam_index=%d)" % cam_index
            self.cap = None

        self.cap_face_active = False
        self.cap_hand_active = False
        self.face_target = 0
        self.hand_target = 0
        self.face_saved = 0
        self.hand_saved = 0
        self.face_every = 2
        self.hand_every = 2
        self.face_name = ""
        self.hand_label = ""
        self.frame_count = 0
        self.show_fps = True
        self._t_prev = time.time()
        self._fps = 0.0
        self.clean_mode = False 
    
    def set_clean_mode(self, clean=True):
        with self.lock:
            self.clean_mode = clean
    
    def capture_current_frame(self):
        if self.cap is None:
            return None, "Kamera tidak tersedia"
        
        ok, frame = self.cap.read()
        if not ok:
            return None, "Gagal membaca frame dari kamera"
        
        return frame, None
    
    def predict_frame(self, frame):
        face_results = []
        hand_results = []

        face_res, face_err = self.face.predict(frame)
        if not face_err and face_res:
            face_results = face_res

        hand_res, hand_err = self.hand.predict(frame)
        if not hand_err and hand_res:
            hand_results = hand_res
        
        return {
            'face_results': face_results,
            'hand_results': hand_results,
            'face_error': face_err,
            'hand_error': hand_err
        }
    
    def start_capture_face(self, name, num=100, every=2):
        with self.lock:
            os.makedirs(os.path.join(DATASET_FACES_DIR, name), exist_ok=True)
            self.face_name = name
            self.face_target = int(num)
            self.face_saved = 0
            self.face_every = max(1, int(every))
            self.cap_face_active = True

    def start_capture_hand(self, label, num=100, every=2):
        with self.lock:
            os.makedirs(os.path.join(DATASET_HANDS_DIR, label), exist_ok=True)
            self.hand_label = label
            self.hand_target = int(num)
            self.hand_saved = 0
            self.hand_every = max(1, int(every))
            self.cap_hand_active = True

    def stop_capture(self):
        with self.lock:
            self.cap_face_active = False
            self.cap_hand_active = False

    def status(self):
        with self.lock:
            return {
                "face": {
                    "active": self.cap_face_active, 
                    "saved": self.face_saved, 
                    "target": self.face_target, 
                    "name": self.face_name, 
                    "error": self.face.error
                },
                "hand": {
                    "active": self.cap_hand_active, 
                    "saved": self.hand_saved, 
                    "target": self.hand_target, 
                    "label": self.hand_label, 
                    "error": self.hand.error
                },
                "fps": self._fps,
                "cam_error": self.cam_error,
            }

    def _encode_frame(self, frame):
        ret, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ret:
            return None
        return buf.tobytes()

    def generate(self):
        if self.cap is None:
            msg = ["ERROR: Kamera tidak bisa dibuka.", "Cek index kamera & izin akses webcam."]
            err = draw_error_frame(msg)
            data = self._encode_frame(err)
            while True:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        
        while True:
            ok, frame = self.cap.read()
            if not ok:
                frame = draw_error_frame(["ERROR: Gagal membaca frame dari kamera."])
            
            self.frame_count += 1

            if not self.clean_mode:
                face_res, face_err = self.face.predict(frame)
                if face_err:
                    put_shadow_text(frame, f"Face model error: {face_err}", (10, 30), scale=0.6, color=(0, 140, 255))
                else:
                    for r in face_res:
                        x1,y1,x2,y2 = r["box"]
                        color = (0,200,0) if r["name"] != "Unknown" else (0,0,255)
                        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                        put_shadow_text(frame, f'{r["name"]} ({r["prob"]*100:.1f}%)', (x1, y1-8), scale=0.6)

                hand_res, hand_err = self.hand.predict(frame)
                if hand_err:
                    put_shadow_text(frame, f"Hand model error: {hand_err}", (10, 55), scale=0.6, color=(0, 140, 255))
                else:
                    for r in hand_res:
                        x1,y1,x2,y2 = r["box"]
                        color = (200,200,0) if r["name"] != "Unknown" else (0,0,255)
                        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                        put_shadow_text(frame, f'Hand: {r["name"]} ({r["prob"]*100:.1f}%)', (x1, y2+20), scale=0.6)

                t_now = time.time()
                dt = max(t_now - self._t_prev, 1e-6)
                self._fps = 0.9*self._fps + 0.1*(1.0/dt)
                self._t_prev = t_now
                put_shadow_text(frame, f"FPS: {self._fps:.1f}", (10, frame.shape[0]-15), scale=0.7, color=(255,255,0))

            with self.lock:
                if self.cap_face_active and self.face_saved < self.face_target and (self.frame_count % self.face_every == 0):
                    aligned = self.face.aligned_face(frame)  # RGB 160x160
                    if aligned is not None:
                        bgr = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)
                        filename = os.path.join(DATASET_FACES_DIR, self.face_name, f"{self.face_name}_{self.face_saved:04d}.png")
                        cv2.imwrite(filename, bgr)
                        self.face_saved += 1
                    if self.face_saved >= self.face_target:
                        self.cap_face_active = False

                if self.cap_hand_active and self.hand_saved < self.hand_target and (self.frame_count % self.hand_every == 0):
                    crop = self.hand.crop_first_hand(frame)  # RGB 224x224
                    if crop is not None:
                        bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                        filename = os.path.join(DATASET_HANDS_DIR, self.hand_label, f"{self.hand_label}_{self.hand_saved:04d}.png")
                        cv2.imwrite(filename, bgr)
                        self.hand_saved += 1
                    if self.hand_saved >= self.hand_target:
                        self.cap_hand_active = False

            data = self._encode_frame(frame)
            if data is None:
                continue
            
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
