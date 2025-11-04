from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime, time
import mediapipe as mp
import os
import threading
from model_trainer import FaceTrainer, HandGestureTrainer
from webcam_attendance import VideoProcessor
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

if not os.path.exists('models'):
    os.makedirs('models')

class ProgramStudi(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nama = db.Column(db.String(100), nullable=False)
    kode = db.Column(db.String(10), nullable=False, unique=True)
    mahasiswa = db.relationship('Mahasiswa', backref='program_studi', lazy=True)

class Mahasiswa(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nim = db.Column(db.String(20), nullable=False, unique=True)
    nama = db.Column(db.String(100), nullable=False)
    program_studi_id = db.Column(db.Integer, db.ForeignKey('program_studi.id'), nullable=False)
    face_trained = db.Column(db.Boolean, default=False)
    hand_trained = db.Column(db.Boolean, default=False)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='mahasiswa')  # mahasiswa, admin
    mahasiswa_id = db.Column(db.Integer, db.ForeignKey('mahasiswa.id'), nullable=True)
    mahasiswa = db.relationship('Mahasiswa', backref=db.backref('user', uselist=False))

class MataKuliah(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nama = db.Column(db.String(100), nullable=False)
    kode = db.Column(db.String(10), nullable=False, unique=True)
    sks = db.Column(db.Integer, nullable=False)
    program_studi_id = db.Column(db.Integer, db.ForeignKey('program_studi.id'), nullable=False)
    jumlah_pertemuan = db.Column(db.Integer, nullable=False, default=16)
    waktu_mulai = db.Column(db.Time, nullable=False, default=time(8, 0))  # Jam mulai kuliah
    waktu_selesai = db.Column(db.Time, nullable=False, default=time(10, 0))  # Jam selesai kuliah
    batas_waktu = db.Column(db.Time, nullable=False, default=time(8, 30))  # Batas waktu absensi
    
    program_studi = db.relationship('ProgramStudi', backref=db.backref('mata_kuliah_list', lazy=True))

class Absensi(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    mahasiswa_id = db.Column(db.Integer, db.ForeignKey('mahasiswa.id'), nullable=False)
    mata_kuliah_id = db.Column(db.Integer, db.ForeignKey('mata_kuliah.id'), nullable=False)
    tanggal = db.Column(db.Date, nullable=False)
    waktu = db.Column(db.Time, nullable=False)
    status = db.Column(db.String(20), nullable=False, default='Hadir')
    pertemuan = db.Column(db.Integer, nullable=False, default=1)  # Field baru untuk nomor pertemuan
    metode = db.Column(db.String(20), nullable=False, default='Manual')
    foto_path = db.Column(db.String(255), nullable=True)
    
    mahasiswa = db.relationship('Mahasiswa', backref=db.backref('absensi_list', lazy=True))
    mata_kuliah = db.relationship('MataKuliah', backref=db.backref('absensi_list', lazy=True))

face_model = None
hand_model = None
label_encoder = None
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

vp = None
def get_vp():
    global vp
    if vp is None:
        vp = VideoProcessor(cam_index=0)
    return vp

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def mahasiswa_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'mahasiswa':
            flash('Akses ditolak. Hanya untuk mahasiswa.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            session['mahasiswa_id'] = user.mahasiswa_id
            
            if user.role == 'admin':
                return redirect(url_for('dashboard'))
            else:
                return redirect(url_for('mahasiswa_dashboard'))
        else:
            flash('Username atau password salah!', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Anda telah logout.', 'success')
    return redirect(url_for('login'))

@app.route('/admin/create_mahasiswa_users')
@login_required
def admin_create_mahasiswa_users():
    if session.get('role') != 'admin':
        flash('Akses ditolak. Hanya untuk admin.', 'error')
        return redirect(url_for('dashboard'))
    
    try:
        mahasiswa_list = Mahasiswa.query.all()
        users_created = 0
        
        for mhs in mahasiswa_list:
            existing_user = User.query.filter_by(mahasiswa_id=mhs.id).first()
            if not existing_user:
                username = mhs.nim
                password = mhs.nim
                
                user = User(
                    username=username,
                    password_hash=generate_password_hash(password),
                    role='mahasiswa',
                    mahasiswa_id=mhs.id
                )
                db.session.add(user)
                users_created += 1
        
        if users_created > 0:
            db.session.commit()
            flash(f'Berhasil membuat {users_created} akun mahasiswa!', 'success')
        else:
            flash('Semua mahasiswa sudah memiliki akun.', 'info')
            
    except Exception as e:
        db.session.rollback()
        flash(f'Error: {str(e)}', 'error')
    
    return redirect(url_for('mahasiswa'))

@app.route('/')
def index():
    total_mahasiswa = Mahasiswa.query.count()
    total_prodi = ProgramStudi.query.count()
    total_matkul = MataKuliah.query.count()
    total_absensi = Absensi.query.count()
    
    trained_students = Mahasiswa.query.filter(
        (Mahasiswa.face_trained == True) & (Mahasiswa.hand_trained == True)
    ).count()
    
    recent_absensi = Absensi.query.order_by(Absensi.tanggal.desc()).limit(10).all()
    
    face_model_status = os.path.exists('models/face_classifier.pkl')
    hand_model_status = os.path.exists('models/hand_model.h5')
    
    return render_template('dashboard.html',
                         total_mahasiswa=total_mahasiswa,
                         total_prodi=total_prodi,
                         total_matkul=total_matkul,
                         total_absensi=total_absensi,
                         trained_students=trained_students,
                         recent_absensi=recent_absensi,
                         face_model_status=face_model_status,
                         hand_model_status=hand_model_status)

@app.route('/mahasiswa/dashboard')
@mahasiswa_required
def mahasiswa_dashboard():
    mahasiswa_id = session.get('mahasiswa_id')
    if not mahasiswa_id:
        flash('Data mahasiswa tidak ditemukan.', 'error')
        return redirect(url_for('login'))
    
    mahasiswa = Mahasiswa.query.get_or_404(mahasiswa_id)
    
    # Ambil absensi mahasiswa
    absensi_list = Absensi.query.filter_by(mahasiswa_id=mahasiswa_id)\
                               .order_by(Absensi.tanggal.desc())\
                               .limit(10).all()
    
    # Hitung statistik
    total_absensi = Absensi.query.filter_by(mahasiswa_id=mahasiswa_id).count()
    total_hadir = Absensi.query.filter_by(mahasiswa_id=mahasiswa_id, status='Hadir').count()
    total_mata_kuliah = db.session.query(Absensi.mata_kuliah_id)\
                                 .filter_by(mahasiswa_id=mahasiswa_id)\
                                 .distinct().count()
    
    return render_template('mahasiswa_dashboard.html',
                         mahasiswa=mahasiswa,
                         absensi_list=absensi_list,
                         total_absensi=total_absensi,
                         total_hadir=total_hadir,
                         total_mata_kuliah=total_mata_kuliah)

@app.route('/mahasiswa/absensi')
@mahasiswa_required
def mahasiswa_absensi():
    mahasiswa_id = session.get('mahasiswa_id')
    if not mahasiswa_id:
        flash('Data mahasiswa tidak ditemukan.', 'error')
        return redirect(url_for('login'))
    
    absensi_list = Absensi.query.filter_by(mahasiswa_id=mahasiswa_id)\
                               .order_by(Absensi.tanggal.desc())\
                               .all()
    
    return render_template('mahasiswa_absensi.html', absensi_list=absensi_list)

@app.route('/mahasiswa/mata_kuliah')
@mahasiswa_required
def mahasiswa_mata_kuliah():
    mahasiswa_id = session.get('mahasiswa_id')
    if not mahasiswa_id:
        flash('Data mahasiswa tidak ditemukan.', 'error')
        return redirect(url_for('login'))
    
    mahasiswa = Mahasiswa.query.get_or_404(mahasiswa_id)
    
    # Ambil mata kuliah dari program studi yang sama
    mata_kuliah_list = MataKuliah.query.filter_by(program_studi_id=mahasiswa.program_studi_id).all()
    
    # Hitung persentase kehadiran untuk setiap mata kuliah
    for matkul in mata_kuliah_list:
        total_pertemuan = Absensi.query.filter_by(
            mahasiswa_id=mahasiswa_id, 
            mata_kuliah_id=matkul.id
        ).count()
        
        hadir_count = Absensi.query.filter_by(
            mahasiswa_id=mahasiswa_id, 
            mata_kuliah_id=matkul.id,
            status='Hadir'
        ).count()
        
        matkul.total_pertemuan = total_pertemuan
        matkul.hadir_count = hadir_count
        matkul.persentase = (hadir_count / total_pertemuan * 100) if total_pertemuan > 0 else 0
    
    return render_template('mahasiswa_mata_kuliah.html', 
                         mata_kuliah_list=mata_kuliah_list,
                         mahasiswa=mahasiswa)

@app.route('/dashboard')
def dashboard():
    total_mahasiswa = Mahasiswa.query.count()
    total_prodi = ProgramStudi.query.count()
    total_matkul = MataKuliah.query.count()
    total_absensi = Absensi.query.count()
    
    trained_students = Mahasiswa.query.filter(
        (Mahasiswa.face_trained == True) & (Mahasiswa.hand_trained == True)
    ).count()
    
    recent_absensi = Absensi.query.order_by(Absensi.tanggal.desc()).limit(10).all()
    
    face_model_status = os.path.exists('models/face_classifier.h5') and os.path.exists('models/face_label_encoder.pkl')
    hand_model_status = os.path.exists('models/hand_gesture_model.h5') and os.path.exists('models/hand_label_encoder.pkl')
    
    return render_template('dashboard.html', 
                         total_mahasiswa=total_mahasiswa,
                         total_prodi=total_prodi,
                         total_matkul=total_matkul,
                         total_absensi=total_absensi,
                         trained_students=trained_students,
                         recent_absensi=recent_absensi,
                         face_model_status=face_model_status,
                         hand_model_status=hand_model_status)

@app.route('/program_studi')
def program_studi():
    prodi_list = ProgramStudi.query.all()
    return render_template('program_studi.html', prodi_list=prodi_list)

@app.route('/program_studi/add', methods=['GET', 'POST'])
def add_program_studi():
    if request.method == 'POST':
        nama = request.form['nama']
        kode = request.form['kode']
        
        prodi = ProgramStudi(nama=nama, kode=kode)
        db.session.add(prodi)
        db.session.commit()
        
        return redirect(url_for('program_studi'))
    
    return render_template('add_program_studi.html')

@app.route('/program_studi/<int:prodi_id>/mahasiswa')
def list_mahasiswa_by_prodi(prodi_id):
    prodi = ProgramStudi.query.get_or_404(prodi_id)
    mahasiswa_list = Mahasiswa.query.filter_by(program_studi_id=prodi_id).all()
    
    for mhs in mahasiswa_list:
        face_dataset_path = f"datasets/faces/{mhs.nim}_{mhs.nama}"
        hand_dataset_path = f"datasets/hands/{mhs.nim}_{mhs.nama}"
        
        mhs.face_dataset_count = 0
        mhs.hand_dataset_count = 0
        
        if os.path.exists(face_dataset_path):
            mhs.face_dataset_count = len([f for f in os.listdir(face_dataset_path) 
                                        if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        if os.path.exists(hand_dataset_path):
            mhs.hand_dataset_count = len([f for f in os.listdir(hand_dataset_path) 
                                        if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    return render_template('mahasiswa_by_prodi.html', 
                         mahasiswa_list=mahasiswa_list, 
                         prodi=prodi)

@app.route('/mahasiswa')
def mahasiswa():
    mahasiswa_list = Mahasiswa.query.all()
    
    for mhs in mahasiswa_list:
        face_dataset_path = f"datasets/faces/{mhs.nim}_{mhs.nama}"
        hand_dataset_path = f"datasets/hands/{mhs.nim}_{mhs.nama}"
        
        mhs.face_dataset_count = 0
        mhs.hand_dataset_count = 0
        
        if os.path.exists(face_dataset_path):
            mhs.face_dataset_count = len([f for f in os.listdir(face_dataset_path) 
                                        if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        if os.path.exists(hand_dataset_path):
            mhs.hand_dataset_count = len([f for f in os.listdir(hand_dataset_path) 
                                        if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    return render_template('mahasiswa.html', mahasiswa_list=mahasiswa_list)

@app.route('/mahasiswa/add', methods=['GET', 'POST'])
@app.route('/mahasiswa/add/<int:prodi_id>', methods=['GET', 'POST'])
def add_mahasiswa(prodi_id=None):
    if request.method == 'POST':
        nim = request.form['nim']
        nama = request.form['nama']
        program_studi_id = request.form['program_studi_id']
        
        mahasiswa = Mahasiswa(nim=nim, nama=nama, program_studi_id=program_studi_id)
        db.session.add(mahasiswa)
        db.session.commit()
        
        # Redirect ke halaman mahasiswa_by_prodi sesuai program studi yang dipilih
        return redirect(url_for('list_mahasiswa_by_prodi', prodi_id=program_studi_id))
    
    prodi_list = ProgramStudi.query.all()
    return render_template('add_mahasiswa.html', prodi_list=prodi_list, selected_prodi_id=prodi_id)

@app.route('/mahasiswa/<int:mahasiswa_id>/edit', methods=['GET', 'POST'])
@app.route('/mahasiswa/<int:mahasiswa_id>/edit/<int:prodi_id>', methods=['GET', 'POST'])
def edit_mahasiswa(mahasiswa_id, prodi_id=None):
    mahasiswa = Mahasiswa.query.get_or_404(mahasiswa_id)
    
    if request.method == 'POST':
        mahasiswa.nim = request.form['nim']
        mahasiswa.nama = request.form['nama']
        mahasiswa.program_studi_id = request.form['program_studi_id']
        
        db.session.commit()
        
        # Redirect ke halaman yang sesuai berdasarkan prodi_id
        if prodi_id:
            return redirect(url_for('list_mahasiswa_by_prodi', prodi_id=prodi_id))
        else:
            return redirect(url_for('mahasiswa'))
    
    prodi_list = ProgramStudi.query.all()
    return render_template('edit_mahasiswa.html', mahasiswa=mahasiswa, prodi_list=prodi_list, selected_prodi_id=prodi_id)

@app.route('/mahasiswa/<int:mahasiswa_id>/delete', methods=['POST'])
def delete_mahasiswa(mahasiswa_id):
    mahasiswa = Mahasiswa.query.get_or_404(mahasiswa_id)
    
    # Hapus folder dataset jika ada
    face_dataset_path = f"datasets/faces/{mahasiswa.nim}_{mahasiswa.nama}"
    hand_dataset_path = f"datasets/hands/{mahasiswa.nim}_{mahasiswa.nama}"
    
    import shutil
    if os.path.exists(face_dataset_path):
        shutil.rmtree(face_dataset_path)
    if os.path.exists(hand_dataset_path):
        shutil.rmtree(hand_dataset_path)
    
    db.session.delete(mahasiswa)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Mahasiswa berhasil dihapus'})

@app.route('/mata_kuliah')
def mata_kuliah():
    matkul_list = MataKuliah.query.all()
    return render_template('mata_kuliah.html', matkul_list=matkul_list)

@app.route('/mata_kuliah/add', methods=['GET', 'POST'])
def add_mata_kuliah():
    if request.method == 'POST':
        nama = request.form['nama']
        kode = request.form['kode']
        sks = int(request.form['sks'])
        program_studi_id = int(request.form['program_studi_id'])
        jumlah_pertemuan = int(request.form['jumlah_pertemuan'])
        waktu_mulai_str = request.form['waktu_mulai']
        batas_waktu_str = request.form['batas_waktu']
        
        # Konversi string waktu ke objek time
        waktu_mulai = datetime.strptime(waktu_mulai_str, '%H:%M').time()
        batas_waktu = datetime.strptime(batas_waktu_str, '%H:%M').time()
        
        # Hitung waktu selesai otomatis berdasarkan SKS (1 SKS = 50 menit)
        from datetime import timedelta
        waktu_mulai_dt = datetime.combine(datetime.today(), waktu_mulai)
        durasi_menit = sks * 50  # 1 SKS = 50 menit
        waktu_selesai_dt = waktu_mulai_dt + timedelta(minutes=durasi_menit)
        waktu_selesai = waktu_selesai_dt.time()
        
        mata_kuliah = MataKuliah(
            nama=nama, 
            kode=kode, 
            sks=sks, 
            program_studi_id=program_studi_id,
            jumlah_pertemuan=jumlah_pertemuan,
            waktu_mulai=waktu_mulai,
            waktu_selesai=waktu_selesai,
            batas_waktu=batas_waktu
        )
        
        try:
            db.session.add(mata_kuliah)
            db.session.commit()
            flash('Mata kuliah berhasil ditambahkan!', 'success')
            return redirect(url_for('mata_kuliah'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error: {str(e)}', 'error')
    
    program_studi_list = ProgramStudi.query.all()
    return render_template('add_mata_kuliah.html', program_studi_list=program_studi_list)

@app.route('/absensi')
def absensi():
    absensi_list = Absensi.query.all()
    return render_template('absensi.html', absensi_list=absensi_list)

@app.route('/training/mahasiswa/<int:mahasiswa_id>/training')
def training_page(mahasiswa_id):
    mahasiswa = Mahasiswa.query.get_or_404(mahasiswa_id)
    return render_template('training.html', mahasiswa=mahasiswa)

training_progress = {
    'status': 'idle',
    'message': '',
    'progress': 0
}

@app.route('/collect_dataset/<int:mahasiswa_id>', methods=['POST'])
def collect_dataset(mahasiswa_id):
    global training_progress
    
    mahasiswa = Mahasiswa.query.get_or_404(mahasiswa_id)
    
    def collect_data():
        global training_progress
        try:
            training_progress['status'] = 'collecting'
            training_progress['message'] = 'Mengumpulkan dataset...'
            training_progress['progress'] = 0
            
            from dataset_collector import DatasetCollector
            collector = DatasetCollector()
            
            face_count, hand_count = collector.collect_both_datasets(
                mahasiswa.nim, mahasiswa.nama, target_count=100
            )
            
            training_progress['progress'] = 100
            training_progress['status'] = 'completed'
            training_progress['message'] = f'Dataset berhasil dikumpulkan: {face_count} wajah, {hand_count} tangan'
            
        except Exception as e:
            training_progress['status'] = 'error'
            training_progress['message'] = f'Error: {str(e)}'
    
    thread = threading.Thread(target=collect_data)
    thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/train_models', methods=['POST'])
def train_models():
    global training_progress
    
    def train():
        global training_progress
        try:
            training_progress['status'] = 'training'
            training_progress['message'] = 'Training model...'
            training_progress['progress'] = 0
            
            from model_trainer import ModelTrainer
            trainer = ModelTrainer()
            
            success = trainer.train_both_models(
                'datasets/faces',
                'datasets/hands'
            )
            
            if success:
                training_progress['progress'] = 100
                training_progress['status'] = 'completed'
                training_progress['message'] = 'Model berhasil ditraining!'
                
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
    
    thread = threading.Thread(target=train)
    thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/training/train-all-models', methods=['GET', 'POST'])
def train_all_models():
    if request.method == 'GET':
        return jsonify({
            'success': False, 
            'message': 'Gunakan POST method atau tombol training di halaman mahasiswa'
        })
    
    try:
        mahasiswa_list = Mahasiswa.query.all()
        eligible_students = []
        
        for mhs in mahasiswa_list:
            face_dir = f'datasets/faces/{mhs.nim}_{mhs.nama}'
            hand_dir = f'datasets/hands/{mhs.nim}_{mhs.nama}'
            
            face_count = 0
            hand_count = 0
            
            if os.path.exists(face_dir):
                face_count = len([f for f in os.listdir(face_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
            if os.path.exists(hand_dir):
                hand_count = len([f for f in os.listdir(hand_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
            
            if face_count >= 100 and hand_count >= 100:
                eligible_students.append(mhs)
        
        if not eligible_students:
            return jsonify({'success': False, 'message': 'Tidak ada mahasiswa dengan dataset lengkap (minimal 100 gambar wajah dan tangan)'})
        
        def train_all():
            global training_progress
            training_progress = {
                'progress': 0,
                'status': 'training_all',
                'message': f'Memulai training semua model untuk {len(eligible_students)} mahasiswa...'
            }
            
            try:
                training_progress['message'] = 'Training model wajah untuk semua mahasiswa...'
                training_progress['progress'] = 25
                
                print("Starting face training...")
                face_trainer = FaceTrainer()
                face_history, face_accuracy = face_trainer.train(
                    dataset_path="datasets/faces",
                    model_save_path="models"
                )
                print(f"Face training completed with accuracy: {face_accuracy}")
                
                training_progress['message'] = 'Training model tangan untuk semua mahasiswa...'
                training_progress['progress'] = 75
                
                print("Starting hand training...")
                hand_trainer = HandGestureTrainer()
                hand_history, hand_accuracy = hand_trainer.train(
                    dataset_path="datasets/hands",
                    model_save_path="models"
                )
                print(f"Hand training completed with accuracy: {hand_accuracy}")
                
            except Exception as e:
                print(f"Detailed training error: {e}")
                import traceback
                traceback.print_exc()
                training_progress['status'] = 'error'
                training_progress['message'] = f'Error training all models: {str(e)}'
                print(f"Training error details: {e}")
        
        thread = threading.Thread(target=train_all)
        thread.start()
        
        return jsonify({'success': True, 'message': f'Training semua model dimulai untuk {len(eligible_students)} mahasiswa'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/training_progress')
def get_training_progress():
    global training_progress
    return jsonify(training_progress)

training_progress = {
    'progress': 0,
    'status': 'idle',
    'message': 'Belum ada training yang berjalan'
}

@app.route("/attendance_webcam")
def attendance_webcam():
    get_vp().set_clean_mode(True)
    
    status = get_vp().status()
    
    face_model_status = not status['face']['error']
    hand_model_status = not status['hand']['error']
    
    mata_kuliah_list = MataKuliah.query.all()
    
    return render_template("attendance_webcam.html", 
                         status=status,
                         face_model_status=face_model_status,
                         hand_model_status=hand_model_status,
                         mata_kuliah_list=mata_kuliah_list)

# Route baru untuk absensi webcam dengan mata kuliah tertentu
@app.route("/mata_kuliah/<int:mata_kuliah_id>/attendance_webcam")
def attendance_webcam_mata_kuliah(mata_kuliah_id):
    get_vp().set_clean_mode(True)
    
    status = get_vp().status()
    
    face_model_status = not status['face']['error']
    hand_model_status = not status['hand']['error']
    
    mata_kuliah = MataKuliah.query.get_or_404(mata_kuliah_id)
    
    # Hitung pertemuan berikutnya berdasarkan absensi terakhir
    last_absensi = Absensi.query.filter_by(mata_kuliah_id=mata_kuliah_id)\
                                .order_by(Absensi.tanggal.desc())\
                                .first()
    
    if last_absensi:
        from datetime import date
        if last_absensi.tanggal == date.today():
            next_pertemuan = last_absensi.pertemuan if hasattr(last_absensi, 'pertemuan') else 1
        else:
            next_pertemuan = (last_absensi.pertemuan if hasattr(last_absensi, 'pertemuan') else 0) + 1
    else:
        next_pertemuan = 1
    
    # Pastikan tidak melebihi jumlah pertemuan maksimal
    if next_pertemuan > mata_kuliah.jumlah_pertemuan:
        next_pertemuan = mata_kuliah.jumlah_pertemuan
    
    return render_template("attendance_webcam.html", 
                         status=status,
                         current_pertemuan=next_pertemuan,
                         face_model_status=face_model_status,
                         hand_model_status=hand_model_status,
                         mata_kuliah=mata_kuliah,
                         mata_kuliah_id=mata_kuliah_id)

@app.route("/capture_and_predict", methods=["POST"])
def capture_and_predict():
    try:
        data = request.get_json()
        mata_kuliah_id = data.get('mata_kuliah_id')
        
        if not mata_kuliah_id:
            return jsonify({
                'success': False,
                'message': 'Mata kuliah harus dipilih terlebih dahulu'
            })
        
        mata_kuliah = MataKuliah.query.get(mata_kuliah_id)
        if not mata_kuliah:
            return jsonify({
                'success': False,
                'message': 'Mata kuliah tidak valid'
            })
        
        vp = get_vp()
        
        frame, error = vp.capture_current_frame()
        if error:
            return jsonify({
                'success': False,
                'message': error
            })
        
        results = vp.predict_frame(frame)
        
        face_detected = False
        hand_detected = False
        face_name = None
        hand_name = None
        
        if results['face_results'] and len(results['face_results']) > 0:
            face_result = results['face_results'][0]
            if face_result['name'] != 'Unknown':
                face_detected = True
                face_name = face_result['name']
        
        if results['hand_results'] and len(results['hand_results']) > 0:
            hand_result = results['hand_results'][0]
            if hand_result['name'] != 'Unknown':
                hand_detected = True
                hand_name = hand_result['name']
        
        attendance_valid = False
        validation_message = ""
        
        if not face_detected and not hand_detected:
            validation_message = "Tidak dapat mengabsen: Wajah dan tangan tidak terdeteksi"
        elif not face_detected:
            validation_message = "Tidak dapat mengabsen: Wajah tidak terdeteksi"
        elif not hand_detected:
            validation_message = "Tidak dapat mengabsen: Tangan tidak terdeteksi"
        elif face_name != hand_name:
            validation_message = f"Tidak dapat mengabsen: Wajah ({face_name}) & tangan ({hand_name}) tidak cocok"
        else:
            attendance_valid = True
            
            # Tentukan status berdasarkan waktu absensi
            current_time = datetime.now().time()
            if current_time <= mata_kuliah.batas_waktu:
                status_absensi = 'Hadir'
                validation_message = f"Absensi berhasil: {face_name} - Status Hadir"
            else:
                status_absensi = 'Alpha'
                validation_message = f"Absensi terlambat: {face_name} - Status Alpha (melewati batas waktu {mata_kuliah.batas_waktu.strftime('%H:%M')})"
            
            try:
                mahasiswa = get_mahasiswa_by_name(face_name)
                if mahasiswa:
                    current_date = datetime.now().date()
                    
                    existing_attendance = check_existing_attendance(
                        mahasiswa.id, mata_kuliah_id, current_date
                    )
                    
                    if existing_attendance:
                        return jsonify({
                            'success': False,
                            'message': f'{face_name} sudah melakukan absensi untuk mata kuliah ini hari ini'
                        })
                    
                    # Hitung pertemuan berikutnya
                    last_absensi = Absensi.query.filter_by(mata_kuliah_id=mata_kuliah_id)\
                                                .order_by(Absensi.tanggal.desc())\
                                                .first()
                    
                    if last_absensi:
                        from datetime import date
                        if last_absensi.tanggal == date.today():
                            current_pertemuan = last_absensi.pertemuan if hasattr(last_absensi, 'pertemuan') else 1
                        else:
                            current_pertemuan = (last_absensi.pertemuan if hasattr(last_absensi, 'pertemuan') else 0) + 1
                    else:
                        current_pertemuan = 1
                    
                    # Pastikan tidak melebihi jumlah pertemuan maksimal
                    if current_pertemuan > mata_kuliah.jumlah_pertemuan:
                        current_pertemuan = mata_kuliah.jumlah_pertemuan
                    
                    new_attendance = Absensi(
                        mahasiswa_id=mahasiswa.id,
                        mata_kuliah_id=mata_kuliah_id,
                        tanggal=current_date,
                        waktu=datetime.now().time(),
                        status=status_absensi,
                        pertemuan=current_pertemuan,
                        metode='Webcam'
                    )
                    
                    db.session.add(new_attendance)
                    db.session.commit()
                    
                    return jsonify({
                        'success': True,
                        'message': validation_message,
                        'mahasiswa': face_name,
                        'mata_kuliah': mata_kuliah.nama,
                        'waktu': datetime.now().strftime('%H:%M:%S'),
                        'status': status_absensi,
                        'batas_waktu': mata_kuliah.batas_waktu.strftime('%H:%M')
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': f'Mahasiswa {face_name} tidak ditemukan dalam database'
                    })
                    
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': f'Error menyimpan absensi: {str(e)}'
                })
        
        return jsonify({
            'success': attendance_valid,
            'message': validation_message,
            'face_detected': face_detected,
            'hand_detected': hand_detected,
            'face_name': face_name,
            'hand_name': hand_name
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route("/video_feed")
def video_feed():
    return Response(get_vp().generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.post("/start_capture_face")
def start_capture_face():
    data = request.get_json()
    name = data.get("name", "unknown")
    num = data.get("num", 100)
    every = data.get("every", 2)
    get_vp().start_capture_face(name, num, every)
    return jsonify({"status": "started", "name": name, "num": num})

@app.post("/start_capture_hand")
def start_capture_hand():
    data = request.get_json()
    label = data.get("label", "unknown")
    num = data.get("num", 100)
    every = data.get("every", 2)
    get_vp().start_capture_hand(label, num, every)
    return jsonify({"status": "started", "label": label, "num": num})

@app.post("/stop_capture")
def stop_capture():
    get_vp().stop_capture()
    return jsonify({"status": "stopped"})

def get_mahasiswa_by_name(name):
    try:
        if '_' in name:
            nim, nama = name.split('_', 1)
            mahasiswa = Mahasiswa.query.filter_by(nim=nim, nama=nama).first()
        else:
            mahasiswa = Mahasiswa.query.filter_by(nama=name).first()
        return mahasiswa
    except Exception as e:
        print(f"Error finding mahasiswa: {e}")
        return None

def check_existing_attendance(mahasiswa_id, mata_kuliah_id, tanggal):
    return Absensi.query.filter_by(
        mahasiswa_id=mahasiswa_id,
        mata_kuliah_id=mata_kuliah_id,
        tanggal=tanggal
    ).first()

@app.route('/mata_kuliah/<int:mata_kuliah_id>/input_absensi', methods=['GET'])
def input_absensi(mata_kuliah_id):
    mata_kuliah = MataKuliah.query.get_or_404(mata_kuliah_id)
    
    # Ambil HANYA mahasiswa dari program studi yang SAMA dengan mata kuliah
    mahasiswa_list = Mahasiswa.query.filter_by(program_studi_id=mata_kuliah.program_studi_id).all()
    
    # Ambil riwayat absensi untuk mata kuliah ini, HANYA untuk mahasiswa dari program studi yang sama
    absensi_list = Absensi.query.join(Mahasiswa).filter(
        Absensi.mata_kuliah_id == mata_kuliah_id,
        Mahasiswa.program_studi_id == mata_kuliah.program_studi_id
    ).order_by(Absensi.tanggal.desc(), Absensi.waktu.desc()).all()
    
    # Hitung pertemuan saat ini berdasarkan tanggal unik absensi
    tanggal_unik = db.session.query(Absensi.tanggal).filter_by(mata_kuliah_id=mata_kuliah_id).distinct().count()
    current_pertemuan = tanggal_unik + 1
    
    return render_template('input_absensi.html', 
                         mata_kuliah=mata_kuliah, 
                         mahasiswa_list=mahasiswa_list,
                         absensi_list=absensi_list,
                         current_pertemuan=current_pertemuan)

@app.route('/mata_kuliah/<int:mata_kuliah_id>/absensi')
def absensi_mata_kuliah(mata_kuliah_id):
    mata_kuliah = MataKuliah.query.get_or_404(mata_kuliah_id)
    
    # Ambil semua mahasiswa dari program studi yang sama
    mahasiswa_list = Mahasiswa.query.filter_by(program_studi_id=mata_kuliah.program_studi_id).all()
    
    # Ambil data absensi untuk mata kuliah ini
    absensi_data = {}
    for mahasiswa in mahasiswa_list:
        absensi_mahasiswa = Absensi.query.filter_by(
            mahasiswa_id=mahasiswa.id, 
            mata_kuliah_id=mata_kuliah_id
        ).order_by(Absensi.tanggal).all()
        absensi_data[mahasiswa.id] = absensi_mahasiswa
    
    # Buat list pertemuan (1 sampai jumlah_pertemuan)
    pertemuan_list = list(range(1, mata_kuliah.jumlah_pertemuan + 1))
    
    return render_template('absensi_mata_kuliah.html', 
                         mata_kuliah=mata_kuliah, 
                         mahasiswa_list=mahasiswa_list,
                         absensi_data=absensi_data,
                         pertemuan_list=pertemuan_list)

@app.route('/program_studi/<int:program_studi_id>/riwayat_absensi')
def riwayat_absensi_prodi(program_studi_id):
    program_studi = ProgramStudi.query.get_or_404(program_studi_id)
    
    # Ambil semua mata kuliah dari program studi ini
    mata_kuliah_list = MataKuliah.query.filter_by(program_studi_id=program_studi_id).all()
    
    # Ambil semua absensi dari mata kuliah di program studi ini
    absensi_list = db.session.query(Absensi)\
                            .join(MataKuliah)\
                            .filter(MataKuliah.program_studi_id == program_studi_id)\
                            .order_by(Absensi.tanggal.desc(), Absensi.waktu.desc()).all()
    
    return render_template('riwayat_absensi_prodi.html', 
                         program_studi=program_studi,
                         mata_kuliah_list=mata_kuliah_list,
                         absensi_list=absensi_list)

@app.route('/fix_attendance_data')
def fix_attendance_data():
    from sqlalchemy import or_
    
    # Cari data absensi tanpa field pertemuan atau pertemuan = 0
    old_records = Absensi.query.filter(
        or_(Absensi.pertemuan == None, Absensi.pertemuan == 0)
    ).all()
    
    for record in old_records:
        record.pertemuan = 1
    
    db.session.commit()
    return f"Berhasil update {len(old_records)} data absensi lama"

@app.route('/debug_absensi')
def debug_absensi():
    absensi_data = Absensi.query.all()
    result = []
    for absen in absensi_data:
        result.append({
            'id': absen.id,
            'mahasiswa_id': absen.mahasiswa_id,
            'nim': absen.mahasiswa.nim if absen.mahasiswa else 'Unknown',
            'nama': absen.mahasiswa.nama if absen.mahasiswa else 'Unknown',
            'mata_kuliah': absen.mata_kuliah.nama if absen.mata_kuliah else 'Unknown',
            'status': absen.status,
            'pertemuan': absen.pertemuan,
            'metode': absen.metode,
            'tanggal': absen.tanggal.strftime('%Y-%m-%d') if absen.tanggal else 'No date',
            'waktu': absen.waktu.strftime('%H:%M') if absen.waktu else 'No time'
        })
    return f"<pre>{result}</pre>"

@app.route('/debug_template/<int:mata_kuliah_id>')
def debug_template(mata_kuliah_id):
    mata_kuliah = MataKuliah.query.get_or_404(mata_kuliah_id)
    
    # Ambil semua mahasiswa dari program studi yang sama
    mahasiswa_list = Mahasiswa.query.filter_by(program_studi_id=mata_kuliah.program_studi_id).all()
    
    # Ambil data absensi untuk mata kuliah ini
    absensi_data = {}
    for mahasiswa in mahasiswa_list:
        absensi_mahasiswa = Absensi.query.filter_by(
            mahasiswa_id=mahasiswa.id, 
            mata_kuliah_id=mata_kuliah_id
        ).order_by(Absensi.tanggal).all()
        absensi_data[mahasiswa.id] = absensi_mahasiswa
    
    pertemuan_list = list(range(1, mata_kuliah.jumlah_pertemuan + 1))
    
    result = {
        'mata_kuliah': {
            'id': mata_kuliah.id,
            'nama': mata_kuliah.nama,
            'jumlah_pertemuan': mata_kuliah.jumlah_pertemuan
        },
        'pertemuan_list': pertemuan_list,
        'template_logic_test': {}
    }
    
    # Test logika template untuk setiap mahasiswa
    for mahasiswa in mahasiswa_list:
        absensi_mahasiswa = absensi_data[mahasiswa.id]
        mahasiswa_result = {
            'nama': mahasiswa.nama,
            'nim': mahasiswa.nim,
            'absensi_count': len(absensi_mahasiswa),
            'pertemuan_status': {}
        }
        
        # Test untuk setiap pertemuan
        for pertemuan in pertemuan_list:
            status = 'Alpha'
            found = False
            matching_records = []
            
            for absen in absensi_mahasiswa:
                comparison_result = {
                    'absen_pertemuan': absen.pertemuan,
                    'absen_pertemuan_type': type(absen.pertemuan).__name__,
                    'target_pertemuan': pertemuan,
                    'target_pertemuan_type': type(pertemuan).__name__,
                    'int_comparison': int(absen.pertemuan) == int(pertemuan),
                    'direct_comparison': absen.pertemuan == pertemuan
                }
                matching_records.append(comparison_result)
                
                if int(absen.pertemuan) == int(pertemuan) and not found:
                    status = absen.status
                    found = True
            
            mahasiswa_result['pertemuan_status'][f'pertemuan_{pertemuan}'] = {
                'final_status': status,
                'found_match': found,
                'comparison_details': matching_records
            }
        
        result['template_logic_test'][mahasiswa.id] = mahasiswa_result
    
    return f"<pre>{json.dumps(result, indent=2, ensure_ascii=False)}</pre>"


if __name__ == '__main__':
    with app.app_context():
        # Buat semua tabel
        db.create_all()
        
        # Buat user admin default jika belum ada
        admin_user = User.query.filter_by(username='admin').first()
        if not admin_user:
            admin_user = User(
                username='admin',
                password_hash=generate_password_hash('admin123'),
                role='admin'
            )
            db.session.add(admin_user)
            print("User admin created: username=admin, password=admin123")
        
        # Commit perubahan
        try:
            db.session.commit()
            print("Database initialized successfully!")
        except Exception as e:
            db.session.rollback()
            print(f"Error initializing database: {e}")
    
    app.run(debug=True)