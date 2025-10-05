import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 100

def load_data():
    """Load data dari UCI atau file lokal"""

    # ====== FUNGSI INTERNAL: LOAD DARI UCI ML REPO ======
    def load_from_ucimlrepo():
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=697)
        X, y = ds.data.features, ds.data.targets
        df = pd.concat([X, y], axis=1)
        return df

    # ====== FUNGSI INTERNAL: LOAD DARI FILE LOKAL ======
    def load_from_local_csv(path="data.csv"):
        return pd.read_csv(path, sep=';')  # UCI menggunakan separator ';'

    # ====== LOGIKA PEMILIHAN SUMBER DATA ======
    try:
        df = load_from_ucimlrepo()
        print("[OK] Sumber data: ucimlrepo (online)")
        return df
    except Exception as e:
        print(f"[ERROR] Gagal via ucimlrepo ({e}). Mencoba file lokal...")
        try:
            df = load_from_local_csv("data.csv")
            print("[OK] Sumber data: file lokal data.csv")
            return df
        except Exception as e2:
            print(f"[INFO] File lokal tidak ditemukan. Membuat data sample untuk demo...")
            return create_sample_data()

def create_sample_data():
    """Membuat data sample untuk demo jika data asli tidak tersedia"""

    # ====== KONFIGURASI DATA SAMPLE ======
    np.random.seed(42)  # Seed untuk reproducibility
    n_samples = 1000   # Jumlah sample data

    # ====== PEMBUATAN DATA SAMPLE ======
    data = {
        # Data demografis mahasiswa
        'Marital status': np.random.choice(['Single', 'Married', 'Divorced', 'Widower'], n_samples),
        'Application mode': np.random.choice(['1st phase', '2nd phase', '3rd phase'], n_samples),
        'Application order': np.random.randint(1, 10, n_samples),
        'Course': np.random.choice(['Biofuel Production', 'Civil Engineering', 'Computer Science'], n_samples),
        'Daytime/evening attendance': np.random.choice(['Daytime', 'Evening'], n_samples),

        # Data kualifikasi dan pendidikan
        'Previous qualification': np.random.choice(['Secondary education', 'Higher education'], n_samples),
        'Previous qualification (grade)': np.random.uniform(10, 20, n_samples),
        'Nacionality': np.random.choice(['Portuguese', 'Brazilian', 'Colombian'], n_samples),

        # Data keluarga
        'Mother\'s qualification': np.random.choice(['Secondary education', 'Higher education', 'Basic education'], n_samples),
        'Father\'s qualification': np.random.choice(['Secondary education', 'Higher education', 'Basic education'], n_samples),
        'Mother\'s occupation': np.random.choice(['Student', 'Unemployed', 'Other'], n_samples),
        'Father\'s occupation': np.random.choice(['Student', 'Unemployed', 'Other'], n_samples),

        # Data akademik dan finansial
        'Admission grade': np.random.uniform(100, 200, n_samples),
        'Displaced': np.random.choice(['Yes', 'No'], n_samples),
        'Educational special needs': np.random.choice(['Yes', 'No'], n_samples),
        'Debtor': np.random.choice(['Yes', 'No'], n_samples),
        'Tuition fees up to date': np.random.choice(['Yes', 'No'], n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Scholarship holder': np.random.choice(['Yes', 'No'], n_samples),
        'Age at enrollment': np.random.randint(17, 50, n_samples),
        'International': np.random.choice(['Yes', 'No'], n_samples),

        # Data performa semester 1
        'Curricular units 1st sem (credited)': np.random.randint(0, 20, n_samples),
        'Curricular units 1st sem (enrolled)': np.random.randint(0, 20, n_samples),
        'Curricular units 1st sem (evaluations)': np.random.randint(0, 20, n_samples),
        'Curricular units 1st sem (approved)': np.random.randint(0, 20, n_samples),
        'Curricular units 1st sem (grade)': np.random.uniform(10, 20, n_samples),
        'Curricular units 1st sem (without evaluations)': np.random.randint(0, 5, n_samples),

        # Data performa semester 2
        'Curricular units 2nd sem (credited)': np.random.randint(0, 20, n_samples),
        'Curricular units 2nd sem (enrolled)': np.random.randint(0, 20, n_samples),
        'Curricular units 2nd sem (evaluations)': np.random.randint(0, 20, n_samples),
        'Curricular units 2nd sem (approved)': np.random.randint(0, 20, n_samples),
        'Curricular units 2nd sem (grade)': np.random.uniform(10, 20, n_samples),
        'Curricular units 2nd sem (without evaluations)': np.random.randint(0, 5, n_samples),

        # Data ekonomi makro
        'Unemployment rate': np.random.uniform(5, 25, n_samples),
        'Inflation rate': np.random.uniform(0, 5, n_samples),
        'GDP': np.random.uniform(10000, 50000, n_samples),

        # Target variable
        'Target': np.random.choice(['Dropout', 'Enrolled', 'Graduate'], n_samples)
    }

    return pd.DataFrame(data)

def save_to_csv(df, filename="students_dropout_analysis.csv"):
    """Simpan data ke CSV dengan formatting yang baik"""
    # ====== KONFIGURASI PENYIMPANAN CSV ======
    df.to_csv(filename, index=False, encoding='utf-8')  # Tanpa index, encoding UTF-8
    print(f"[OK] Data tersimpan ke: {filename}")
    return filename

def exploratory_data_analysis(df):
    """Melakukan exploratory data analysis"""
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)

    # ====== ANALISIS INFORMASI DASAR DATASET ======
    print(f"\nINFORMASI DASAR:")
    print(f"   - Ukuran data: {df.shape[0]:,} baris x {df.shape[1]} kolom")
    print(f"   - Memori yang digunakan: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    #====== ANALISIS TIPE DATA ======
    print(f"\nTIPE DATA:")
    print(df.dtypes.value_counts())

    # ====== ANALISIS MISSING VALUES ======
    print(f"\nMISSING VALUES:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   [OK] Tidak ada missing values")
    else:
        print(missing[missing > 0])

    # ====== ANALISIS DISTRIBUSI TARGET ======
    print(f"\nDISTRIBUSI TARGET:")
    target_counts = df['Target'].value_counts()
    print(target_counts)
    print(f"\nPersentase:")
    for target, count in target_counts.items():
        print(f"   - {target}: {count/len(df)*100:.1f}%")

    return df

def statistical_analysis(df):
    """Melakukan analisis statistik"""
    print("\n" + "="*60)
    print("ANALISIS STATISTIK")
    print("="*60)

    # ====== ANALISIS STATISTIK DESKRIPTIF ======
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\nSTATISTIK DESKRIPTIF (Kolom Numerik):")
    print(df[numeric_cols].describe().round(2))

    # ====== ANALISIS KORELASI ======
    print(f"\nANALISIS KORELASI:")
    corr_matrix = df[numeric_cols].corr()

    # ====== IDENTIFIKASI KORELASI TINGGI ======
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:  # Threshold korelasi tinggi
                high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))

    if high_corr:
        print("   Korelasi tinggi (|r| > 0.7):")
        for col1, col2, corr in high_corr:
            print(f"   - {col1} <-> {col2}: {corr:.3f}")
    else:
        print("   [OK] Tidak ada korelasi tinggi yang ditemukan")
    return corr_matrix

def categorical_analysis(df):
    """Analisis untuk kolom kategorikal"""
    print("\n" + "="*60)
    print("ANALISIS KATEGORIKAL")
    print("="*60)

    # ====== IDENTIFIKASI KOLOM KATEGORIKAL ======
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != 'Target']  # Exclude target

    # ====== ANALISIS DISTRIBUSI KATEGORI ======
    print(f"\nDISTRIBUSI KATEGORI:")
    for col in categorical_cols[:5]:  # Show first 5 categorical columns
        print(f"\n{col}:")
        value_counts = df[col].value_counts()
        print(value_counts.head())
        if len(value_counts) > 5:
            print(f"   ... dan {len(value_counts)-5} kategori lainnya")

def create_visualizations(df):
    """Membuat visualisasi untuk analisis"""
    print("\n" + "="*60)
    print("MEMBUAT VISUALISASI")
    print("="*60)

    # Konfigurasi dasar untuk semua grafik
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'

    # ====== CHART 1: PIE CHART DISTRIBUSI TARGET ======
    print("Membuat grafik distribusi target...")
    plt.figure(figsize=(10, 6))
    target_counts = df['Target'].value_counts()
    colors = ['#ff9999', '#66b3ff', '#99ff99']  # Warna untuk Graduate, Dropout, Enrolled
    plt.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', colors=colors)
    plt.title('Distribusi Target (Status Mahasiswa)', fontsize=14, fontweight='bold')
    plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Grafik distribusi target tersimpan")

    # ====== CHART 2: BOX PLOT ANALISIS USIA ======
    print("Membuat grafik analisis usia...")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Target', y='Age at enrollment')
    plt.title('Distribusi Usia berdasarkan Status Mahasiswa', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.savefig('age_by_target.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Grafik analisis usia tersimpan")

    # ====== CHART 3: MULTI-SUBPLOT ANALISIS GRADE ======
    print("Membuat grafik analisis grade...")
    plt.figure(figsize=(15, 5))

    # Subplot 1: Grade Kualifikasi Sebelumnya
    plt.subplot(1, 3, 1)
    sns.boxplot(data=df, x='Target', y='Previous qualification (grade)')
    plt.title('Grade Kualifikasi Sebelumnya')
    plt.xticks(rotation=45)

    # Subplot 2: Grade Penerimaan
    plt.subplot(1, 3, 2)
    sns.boxplot(data=df, x='Target', y='Admission grade')
    plt.title('Grade Penerimaan')
    plt.xticks(rotation=45)

    # Subplot 3: Grade Semester 1
    plt.subplot(1, 3, 3)
    sns.boxplot(data=df, x='Target', y='Curricular units 1st sem (grade)')
    plt.title('Grade Semester 1')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('grade_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Grafik analisis grade tersimpan")

    # ====== CHART 4: BAR CHART DISTRIBUSI GENDER ======
    print("Membuat grafik distribusi gender...")
    plt.figure(figsize=(10, 6))
    gender_counts = df['Gender'].value_counts()
    plt.bar(gender_counts.index, gender_counts.values, color=['#ff9999', '#66b3ff'])  # Pink untuk Female, Blue untuk Male
    plt.title('Distribusi Gender Mahasiswa', fontsize=14, fontweight='bold')
    plt.xlabel('Gender')
    plt.ylabel('Jumlah Mahasiswa')
    plt.savefig('gender_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Grafik distribusi gender tersimpan")

    # ====== CHART 5: MULTI-SUBPLOT PERFORMA AKADEMIK ======
    print("Membuat grafik analisis performa akademik...")
    plt.figure(figsize=(12, 8))

    # Subplot 1: Mata Kuliah Lulus Semester 1
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df, x='Target', y='Curricular units 1st sem (approved)')
    plt.title('Mata Kuliah Lulus Semester 1')
    plt.xticks(rotation=45)

    # Subplot 2: Mata Kuliah Lulus Semester 2
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x='Target', y='Curricular units 2nd sem (approved)')
    plt.title('Mata Kuliah Lulus Semester 2')
    plt.xticks(rotation=45)

    # Subplot 3: Grade Semester 1
    plt.subplot(2, 2, 3)
    sns.boxplot(data=df, x='Target', y='Curricular units 1st sem (grade)')
    plt.title('Grade Semester 1')
    plt.xticks(rotation=45)

    # Subplot 4: Grade Semester 2
    plt.subplot(2, 2, 4)
    sns.boxplot(data=df, x='Target', y='Curricular units 2nd sem (grade)')
    plt.title('Grade Semester 2')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('academic_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Grafik performa akademik tersimpan")
    print("\n[OK] Semua visualisasi berhasil tersimpan sebagai file PNG")

def machine_learning_analysis(df):
    """Melakukan analisis machine learning sederhana"""
    print("\n" + "="*60)
    print("ANALISIS MACHINE LEARNING")
    print("="*60)

    # ====== PREPARASI DATA UNTUK ML ======
    df_ml = df.copy()
    # ====== ENCODING VARIABEL KATEGORIKAL ======
    le_dict = {}
    for col in df_ml.select_dtypes(include=['object']).columns:
        if col != 'Target':
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
            le_dict[col] = le

    # ====== ENCODING TARGET VARIABLE ======
    le_target = LabelEncoder()
    df_ml['Target'] = le_target.fit_transform(df_ml['Target'])

    # ====== PEMISAHAN FEATURES DAN TARGET ======
    X = df_ml.drop('Target', axis=1)
    y = df_ml['Target']

    # ====== SPLIT DATA TRAIN-TEST ======
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ====== TRAINING MODEL RANDOM FOREST ======
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # ====== PREDIKSI DAN EVALUASI ======
    y_pred = rf.predict(X_test)

    # ====== ANALISIS FEATURE IMPORTANCE ======
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTOP 10 FITUR PENTING:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"   {i:2d}. {row['feature']}: {row['importance']:.3f}")

    # ====== PERFORMANCE MODEL ======
    print(f"\nPERFORMANCE MODEL:")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))

    return rf, feature_importance, le_target

def generate_summary_report(df, feature_importance, le_target):
    """Membuat laporan ringkasan analisis"""
    print("\n" + "="*60)
    print("RINGKASAN ANALISIS")
    print("="*60)

    # ====== STATISTIK UTAMA DATASET ======
    total_students = len(df)
    dropout_rate = (df['Target'] == 'Dropout').sum() / total_students * 100
    graduate_rate = (df['Target'] == 'Graduate').sum() / total_students * 100
    enrolled_rate = (df['Target'] == 'Enrolled').sum() / total_students * 100

    print(f"\nSTATISTIK UTAMA:")
    print(f"   - Total mahasiswa: {total_students:,}")
    print(f"   - Tingkat dropout: {dropout_rate:.1f}%")
    print(f"   - Tingkat kelulusan: {graduate_rate:.1f}%")
    print(f"   - Masih terdaftar: {enrolled_rate:.1f}%")

    # ====== INSIGHT UTAMA ======
    print(f"\nINSIGHT UTAMA:")
    print(f"   - Usia rata-rata: {df['Age at enrollment'].mean():.1f} tahun")
    print(f"   - Grade penerimaan rata-rata: {df['Admission grade'].mean():.1f}")
    print(f"   - Persentase mahasiswa internasional: {(df['International'] == 'Yes').sum() / total_students * 100:.1f}%")

    # ====== FAKTOR PENTING BERDASARKAN ML ======
    print(f"\nFAKTOR PENTING (berdasarkan ML):")
    for i, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
        print(f"   {i}. {row['feature']}")

    # ====== REKOMENDASI ======
    print(f"\nREKOMENDASI:")
    print(f"   - Fokus pada faktor-faktor dengan importance tinggi")
    print(f"   - Implementasi program intervensi dini untuk mahasiswa berisiko")
    print(f"   - Monitoring berkala terhadap performa akademik")
    print(f"   - Analisis lebih lanjut untuk faktor spesifik")

def main():
    """Fungsi utama untuk menjalankan seluruh analisis"""
    print("MEMULAI ANALISIS DATA STUDENT DROPOUT/ACADEMIC SUCCESS")
    print("="*60)
    df = load_data()
    csv_filename = save_to_csv(df)
    df = exploratory_data_analysis(df)
    corr_matrix = statistical_analysis(df)
    categorical_analysis(df)
    create_visualizations(df)
    rf_model, feature_importance, le_target = machine_learning_analysis(df)
    generate_summary_report(df, feature_importance, le_target)

    print(f"\nANALISIS SELESAI!")
    print(f"File yang dihasilkan:")
    print(f"   - {csv_filename} - Data dalam format CSV")
    print(f"   - target_distribution.png - Grafik distribusi target")
    print(f"   - age_by_target.png - Analisis usia")
    print(f"   - grade_analysis.png - Analisis grade")
    print(f"   - gender_distribution.png - Distribusi gender")
    print(f"   - academic_performance.png - Performa akademik")

if __name__ == "__main__":
    main()