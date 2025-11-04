from flask import Blueprint, render_template, request, jsonify, redirect, url_for
from dataset_collector import DatasetCollector
from model_trainer import ModelTrainer
from app import db, Mahasiswa
import threading
import os

training_bp = Blueprint('training', __name__)

# Global variables untuk tracking progress
training_progress = {
    'status': 'idle',  # idle, collecting, training, completed, error
    'message': '',
    'progress': 0
}

@training_bp.route('/mahasiswa/<int:mahasiswa_id>/training')
def training_page(mahasiswa_id):
    mahasiswa = Mahasiswa.query.get_or_404(mahasiswa_id)
    return render_template('training.html', mahasiswa=mahasiswa)

@training_bp.route('/collect_dataset/<int:mahasiswa_id>', methods=['POST'])
def collect_dataset(mahasiswa_id):
    global training_progress
    
    mahasiswa = Mahasiswa.query.get_or_404(mahasiswa_id)
    
    def collect_data():
        global training_progress
        try:
            training_progress['status'] = 'collecting'
            training_progress['message'] = 'Mengumpulkan dataset...'
            training_progress['progress'] = 0
            
            collector = DatasetCollector()
            
            # Collect dataset
            face_count, hand_count = collector.collect_both_datasets(
                mahasiswa.nim, mahasiswa.nama, target_count=100
            )
            
            training_progress['progress'] = 100
            training_progress['status'] = 'completed'
            training_progress['message'] = f'Dataset berhasil dikumpulkan: {face_count} wajah, {hand_count} tangan'
            
        except Exception as e:
            training_progress['status'] = 'error'
            training_progress['message'] = f'Error: {str(e)}'
    
    # Mulai collection di background
    thread = threading.Thread(target=collect_data)
    thread.start()
    
    return jsonify({'status': 'started'})

@training_bp.route('/train_models', methods=['POST'])
def train_models():
    global training_progress
    
    def train():
        global training_progress
        try:
            training_progress['status'] = 'training'
            training_progress['message'] = 'Training model...'
            training_progress['progress'] = 0
            
            trainer = ModelTrainer()
            
            # Training kedua model
            success = trainer.train_both_models(
                'datasets/faces',
                'datasets/hands'
            )
            
            if success:
                training_progress['progress'] = 100
                training_progress['status'] = 'completed'
                training_progress['message'] = 'Model berhasil ditraining!'
                
                # Update status mahasiswa
                mahasiswa_list = Mahasiswa.query.all()
                for mahasiswa in mahasiswa_list:
                    mahasiswa.face_trained = True
                    mahasiswa.hand_trained = True
                db.session.commit()
            else:
                training_progress['status'] = 'error'
                training_progress['message'] = 'Gagal training model'
                
        except Exception as e:
            training_progress['status'] = 'error'
            training_progress['message'] = f'Error: {str(e)}'
    
    # Mulai training di background
    thread = threading.Thread(target=train)
    thread.start()
    
    return jsonify({'status': 'started'})

@training_bp.route('/training_progress')
def get_training_progress():
    return jsonify(training_progress)