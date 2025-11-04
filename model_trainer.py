import os
import numpy as np
import cv2
import pickle
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torch.nn as nn
from PIL import Image
import streamlit as st
from datetime import datetime

class FaceTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=0, 
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], 
            factor=0.709, 
            post_process=True,
            device=self.device
        )
        
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        self.classifier = None
        self.label_encoder = LabelEncoder()
        
    def load_face_dataset(self, dataset_path="dataset/face"):
        print("Loading face dataset...")
        
        embeddings = []
        labels = []
        
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path {dataset_path} does not exist!")
        
        classes = [d for d in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, d))]
        
        if len(classes) == 0:
            raise ValueError("No classes found in dataset!")
        
        print(f"Found {len(classes)} classes: {classes}")
        
        for class_name in classes:
            class_path = os.path.join(dataset_path, class_name)
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"Processing class '{class_name}': {len(image_files)} images")
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                
                try:
                    img = Image.open(img_path).convert('RGB')
                    
                    img_cropped = self.mtcnn(img)
                    
                    if img_cropped is not None:
                        img_cropped = img_cropped.unsqueeze(0).to(self.device)
                        
                        with torch.no_grad():
                            embedding = self.facenet(img_cropped).cpu().numpy().flatten()
                        
                        embeddings.append(embedding)
                        labels.append(class_name)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
        
        if len(embeddings) == 0:
            raise ValueError("No valid face embeddings extracted!")
        
        embeddings = np.array(embeddings)
        labels = np.array(labels)
        
        print(f"Extracted {len(embeddings)} embeddings with shape {embeddings.shape}")
        return embeddings, labels
    
    def build_classifier(self, input_dim, num_classes):
        model = Sequential([
            Dense(512, activation='relu', input_shape=(input_dim,)),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, dataset_path="dataset/face", model_save_path="models"):
        print("Starting face recognition training...")
        
        os.makedirs(model_save_path, exist_ok=True)
        
        embeddings, labels = self.load_face_dataset(dataset_path)
        
        encoded_labels = self.label_encoder.fit_transform(labels)
        num_classes = len(self.label_encoder.classes_)
        
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {self.label_encoder.classes_}")
        
        categorical_labels = to_categorical(encoded_labels, num_classes)
        
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, categorical_labels, 
            test_size=0.2, 
            random_state=42, 
            stratify=encoded_labels
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        self.classifier = self.build_classifier(embeddings.shape[1], num_classes)
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(
                os.path.join(model_save_path, 'face_classifier_best.h5'),
                save_best_only=True,
                monitor='val_accuracy'
            ),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        history = self.classifier.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        test_loss, test_accuracy = self.classifier.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        self.classifier.save(os.path.join(model_save_path, 'face_classifier.h5'))
        
        with open(os.path.join(model_save_path, 'face_label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        y_pred = self.classifier.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        print("\nClassification Report:")
        print(classification_report(
            y_true_classes, y_pred_classes, 
            target_names=self.label_encoder.classes_
        ))
        
        self.plot_training_history(history, model_save_path, 'face')
        
        return history, test_accuracy

    def plot_training_history(self, history, save_path, model_type):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title(f'{model_type.title()} Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title(f'{model_type.title()} Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{model_type}_training_history.png'))
        plt.close()

class HandGestureTrainer:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.input_shape = (224, 224, 3)
        
    def load_hand_dataset(self, dataset_path="dataset/hand"):
        print("Loading hand gesture dataset...")
        
        images = []
        labels = []
        
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path {dataset_path} does not exist!")
        
        classes = [d for d in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, d))]
        
        if len(classes) == 0:
            raise ValueError("No classes found in dataset!")
        
        print(f"Found {len(classes)} classes: {classes}")
        
        for class_name in classes:
            class_path = os.path.join(dataset_path, class_name)
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"Processing class '{class_name}': {len(image_files)} images")
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                
                try:
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    img = img.astype(np.float32) / 255.0
                    
                    images.append(img)
                    labels.append(class_name)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
        
        if len(images) == 0:
            raise ValueError("No valid images loaded!")
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"Loaded {len(images)} images with shape {images.shape}")
        return images, labels
    
    def build_model(self, num_classes):
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        base_model.trainable = False
        
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, dataset_path="dataset/hand", model_save_path="models"):
        print("Starting hand gesture recognition training...")
        
        os.makedirs(model_save_path, exist_ok=True)
        
        images, labels = self.load_hand_dataset(dataset_path)
        
        encoded_labels = self.label_encoder.fit_transform(labels)
        num_classes = len(self.label_encoder.classes_)
        
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {self.label_encoder.classes_}")
        
        categorical_labels = to_categorical(encoded_labels, num_classes)
        
        X_train, X_test, y_train, y_test = train_test_split(
            images, categorical_labels,
            test_size=0.2,
            random_state=42,
            stratify=encoded_labels
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        self.model = self.build_model(num_classes)
        
        print("Model Architecture:")
        self.model.summary()
        
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ModelCheckpoint(
                os.path.join(model_save_path, 'hand_gesture_best.h5'),
                save_best_only=True,
                monitor='val_accuracy'
            ),
            ReduceLROnPlateau(patience=7, factor=0.5)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Starting fine-tuning...")
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        fine_tune_at = 50
        
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001/10),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        fine_tune_epochs = 20
        total_epochs = len(history.history['loss']) + fine_tune_epochs
        
        history_fine = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=total_epochs,
            initial_epoch=len(history.history['loss']),
            callbacks=callbacks,
            verbose=1
        )
        
        for key in history.history.keys():
            history.history[key].extend(history_fine.history[key])
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        self.model.save(os.path.join(model_save_path, 'hand_gesture_model.h5'))
        
        with open(os.path.join(model_save_path, 'hand_label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        print("\nClassification Report:")
        print(classification_report(
            y_true_classes, y_pred_classes,
            target_names=self.label_encoder.classes_
        ))
        
        self.plot_training_history(history, model_save_path, 'hand_gesture')
        
        return history, test_accuracy
    
    def plot_training_history(self, history, save_path, model_type):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title(f'{model_type.replace("_", " ").title()} Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title(f'{model_type.replace("_", " ").title()} Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{model_type}_training_history.png'))
        plt.close()

def main():
    """Main training function"""
    print("=" * 60)
    print("TRAINING FACE RECOGNITION AND HAND GESTURE MODELS")
    print("=" * 60)
    
    # Check if datasets exist
    if not os.path.exists("dataset/face") or not os.path.exists("dataset/hand"):
        print("Error: Dataset folders not found!")
        print("Please run the data collection app first to create datasets.")
        return
    
    # Check if there are classes in datasets
    face_classes = [d for d in os.listdir("dataset/face") 
                   if os.path.isdir(os.path.join("dataset/face", d))]
    hand_classes = [d for d in os.listdir("dataset/hand") 
                   if os.path.isdir(os.path.join("dataset/hand", d))]
    
    if len(face_classes) < 2:
        print("Warning: Need at least 2 face classes for training!")
    
    if len(hand_classes) < 2:
        print("Warning: Need at least 2 hand gesture classes for training!")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Train Face Recognition Model
    if len(face_classes) >= 2:
        print("\n" + "=" * 40)
        print("TRAINING FACE RECOGNITION MODEL")
        print("=" * 40)
        
        try:
            face_trainer = FaceTrainer()
            face_history, face_accuracy = face_trainer.train()
            print(f"✅ Face recognition model trained successfully!")
            print(f"Final accuracy: {face_accuracy:.4f}")
        except Exception as e:
            print(f"❌ Error training face model: {e}")
    else:
        print("⚠️ Skipping face recognition training (insufficient classes)")
    
    # Train Hand Gesture Recognition Model
    if len(hand_classes) >= 2:
        print("\n" + "=" * 40)
        print("TRAINING HAND GESTURE RECOGNITION MODEL")
        print("=" * 40)
        
        try:
            hand_trainer = HandGestureTrainer()
            hand_history, hand_accuracy = hand_trainer.train()
            print(f"✅ Hand gesture model trained successfully!")
            print(f"Final accuracy: {hand_accuracy:.4f}")
        except Exception as e:
            print(f"❌ Error training hand gesture model: {e}")
    else:
        print("⚠️ Skipping hand gesture training (insufficient classes)")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print("Models saved in 'models/' directory:")
    print("- face_classifier.h5 (Face recognition model)")
    print("- face_label_encoder.pkl (Face labels)")
    print("- hand_gesture_model.h5 (Hand gesture model)")
    print("- hand_label_encoder.pkl (Hand gesture labels)")
    print("- Training history plots saved as PNG files")

if __name__ == "__main__":
    main()