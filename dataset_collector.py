import cv2
import mediapipe as mp
import os
import numpy as np
from datetime import datetime
import json

class DatasetCollector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.7
        )
        
    def collect_face_dataset(self, student_id, student_name, target_count=100):
        """Mengumpulkan dataset wajah mahasiswa"""
        face_dir = f"datasets/faces/{student_id}_{student_name}"
        os.makedirs(face_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        count = 0
        
        print(f"Mengumpulkan dataset wajah untuk {student_name}...")
        print("Tekan 'q' untuk berhenti")
        
        while count < target_count:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(frame_rgb)
            
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)
                    
                    face = frame[y:y+h, x:x+w]
                    
                    if face.size > 0:
                        face_resized = cv2.resize(face, (160, 160))
                        
                        filename = f"{face_dir}/face_{count:04d}.jpg"
                        cv2.imwrite(filename, face_resized)
                        count += 1
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"Face: {count}/{target_count}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Face Dataset Collection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Dataset wajah selesai: {count} gambar tersimpan")
        return count
    
    def collect_hand_dataset(self, student_id, student_name, target_count=100):
        """Mengumpulkan dataset tangan mahasiswa"""
        hand_dir = f"datasets/hands/{student_id}_{student_name}"
        os.makedirs(hand_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        count = 0
        
        print(f"Mengumpulkan dataset tangan untuk {student_name}...")
        print("Tekan 'q' untuk berhenti")
        
        while count < target_count:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dapatkan bounding box tangan
                    h, w, _ = frame.shape
                    x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
                    y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
                    
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))

                    margin = 20
                    x_min = max(0, x_min - margin)
                    y_min = max(0, y_min - margin)
                    x_max = min(w, x_max + margin)
                    y_max = min(h, y_max + margin)
                    

                    hand = frame[y_min:y_max, x_min:x_max]
                    
                    if hand.size > 0:

                        hand_resized = cv2.resize(hand, (224, 224))

                        filename = f"{hand_dir}/hand_{count:04d}.jpg"
                        cv2.imwrite(filename, hand_resized)
                        count += 1

                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                        cv2.putText(frame, f"Hand: {count}/{target_count}", 
                                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            cv2.imshow('Hand Dataset Collection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Dataset tangan selesai: {count} gambar tersimpan")
        return count
    
    def collect_both_datasets(self, student_id, student_name, target_count=100):
        """Mengumpulkan dataset wajah dan tangan sekaligus"""
        face_dir = f"datasets/faces/{student_id}_{student_name}"
        hand_dir = f"datasets/hands/{student_id}_{student_name}"
        os.makedirs(face_dir, exist_ok=True)
        os.makedirs(hand_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        face_count = 0
        hand_count = 0
        
        print(f"Mengumpulkan dataset wajah dan tangan untuk {student_name}...")
        print("Tekan 'q' untuk berhenti")
        
        while face_count < target_count or hand_count < target_count:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Deteksi wajah
            if face_count < target_count:
                face_results = self.face_detection.process(frame_rgb)
                if face_results.detections:
                    for detection in face_results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        
                        x = int(bboxC.xmin * iw)
                        y = int(bboxC.ymin * ih)
                        w = int(bboxC.width * iw)
                        h = int(bboxC.height * ih)
                        
                        face = frame[y:y+h, x:x+w]
                        
                        if face.size > 0:
                            face_resized = cv2.resize(face, (160, 160))
                            filename = f"{face_dir}/face_{face_count:04d}.jpg"
                            cv2.imwrite(filename, face_resized)
                            face_count += 1
                            
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Deteksi tangan
            if hand_count < target_count:
                hand_results = self.hands.process(frame_rgb)
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        h, w, _ = frame.shape
                        x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
                        y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
                        
                        x_min, x_max = int(min(x_coords)), int(max(x_coords))
                        y_min, y_max = int(min(y_coords)), int(max(y_coords))
                        
                        margin = 20
                        x_min = max(0, x_min - margin)
                        y_min = max(0, y_min - margin)
                        x_max = min(w, x_max + margin)
                        y_max = min(h, y_max + margin)
                        
                        hand = frame[y_min:y_max, x_min:x_max]
                        
                        if hand.size > 0:
                            hand_resized = cv2.resize(hand, (224, 224))
                            filename = f"{hand_dir}/hand_{hand_count:04d}.jpg"
                            cv2.imwrite(filename, hand_resized)
                            hand_count += 1
                            
                            self.mp_drawing.draw_landmarks(
                                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            
            # Tampilkan status
            cv2.putText(frame, f"Face: {face_count}/{target_count}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Hand: {hand_count}/{target_count}", 
                      (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            cv2.imshow('Dataset Collection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Dataset selesai - Wajah: {face_count}, Tangan: {hand_count}")
        return face_count, hand_count