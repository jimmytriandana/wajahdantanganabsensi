import sys
import os
sys.path.append(os.path.dirname(__file__))

from app import app, db, Mahasiswa, User, ProgramStudi
from werkzeug.security import generate_password_hash

def create_mahasiswa_users():
    with app.app_context():
        try:
            # Cek apakah ada data mahasiswa
            mahasiswa_count = Mahasiswa.query.count()
            print(f"Found {mahasiswa_count} mahasiswa in database")
            
            if mahasiswa_count == 0:
                print("No mahasiswa found. Creating sample data...")
                create_sample_data()
            
            mahasiswa_list = Mahasiswa.query.all()
            
            users_created = 0
            for mhs in mahasiswa_list:
                # Cek apakah user sudah ada
                existing_user = User.query.filter_by(mahasiswa_id=mhs.id).first()
                if not existing_user:
                    # Buat username dari NIM
                    username = mhs.nim
                    # Buat password default
                    password = mhs.nim  # atau password default lain
                    
                    user = User(
                        username=username,
                        password_hash=generate_password_hash(password),
                        role='mahasiswa',
                        mahasiswa_id=mhs.id
                    )
                    db.session.add(user)
                    users_created += 1
                    print(f"Created user for {mhs.nama} - username: {username}, password: {password}")
            
            if users_created > 0:
                db.session.commit()
                print(f"Successfully created {users_created} mahasiswa users!")
            else:
                print("No new users created - all mahasiswa already have accounts")
                
        except Exception as e:
            print(f"Error: {e}")
            db.session.rollback()

def create_sample_data():
    """Create sample program studi and mahasiswa if database is empty"""
    try:
        # Create sample program studi
        prodi = ProgramStudi(
            nama="Teknik Informatika",
            kode="TI"
        )
        db.session.add(prodi)
        db.session.commit()
        
        # Create sample mahasiswa
        mahasiswa1 = Mahasiswa(
            nim="20210001",
            nama="John Doe",
            program_studi_id=prodi.id
        )
        
        mahasiswa2 = Mahasiswa(
            nim="20210002", 
            nama="Jane Smith",
            program_studi_id=prodi.id
        )
        
        db.session.add(mahasiswa1)
        db.session.add(mahasiswa2)
        db.session.commit()
        
        print("Sample data created successfully!")
        
    except Exception as e:
        print(f"Error creating sample data: {e}")
        db.session.rollback()

if __name__ == '__main__':
    create_mahasiswa_users()