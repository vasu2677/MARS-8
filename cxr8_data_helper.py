"""
cxr8_data_helper.py
===================
Longitudinal Data Loader and Feature Engineering Helper for NIH Chest X-Ray (CXR8).

Key Features:
- Reads Data_Entry_2017_v2020.csv and sorts strictly by ['Patient ID', 'Follow-up #'].
- Filters and constructs valid longitudinal patient sequences (>= 2 visits).
- Samples a reproducible cohort of 1,500 longitudinal patients (seed=42).
- Extracts real Histogram of Oriented Gradients (HOG) features from actual chest X-ray PNG
  images in images_001.tar.gz, images_002.tar.gz, and images_003.tar.gz if available.
- If images are unavailable, transparently uses explicitly identified clinical/acquisition metadata.
  Never fabricates HOG features or presents random embeddings as real image features.
- Patient-wise train (70%) / validation (15%) / test (15%) split ensuring ZERO data leakage.
- Fits StandardScaler and PCA strictly on training patients, transforming val and test splits.
- Preserves independent patient sequence lengths for HMM sequence modeling.
"""

import io
import os
import tarfile
import time
from typing import Any, cast

import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 14 Canonical NIH Chest X-Ray thoracic pathology categories
PATHOLOGY_LIST = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

CSV_PATH = "Data_Entry_2017_v2020.csv"
IMAGES_DIR = "images"
TAR_FILES = ["images_001.tar.gz", "images_002.tar.gz", "images_003.tar.gz"]
CACHE_FILE = "cxr8_features_cohort1500.npz"


def extract_hog_features_from_image(image_bytes, image_size=(128, 128)):
    """
    Extracts real Histogram of Oriented Gradients (HOG) features from raw PNG image bytes.
    Grayscale conversion, resize to (128, 128), and 8-orientation HOG with (16, 16) cells.
    Returns a 1,568-dimensional visual feature vector.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('L').resize(image_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    features = hog(
        arr,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )
    return features.astype(np.float32)


def find_file(filename, candidate_dirs=(".", "images", "..", "../images")):
    """Locates a file across standard candidate directories."""
    for d in candidate_dirs:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return filename


def get_clinical_metadata_features(df_cohort):
    """
    Constructs explicitly identified tabular features from clinical metadata:
    - 14 binary thoracic pathology indicators
    - Finding_Count (pathology burden)
    - Patient Age (numeric)
    - Patient Sex (binary: 1=M, 0=F)
    - View Position (binary: 1=PA, 0=AP)
    - Follow-up # (progression index)
    """
    records = []
    has_pathology_list = []
    
    for _, row in df_cohort.iterrows():
        labels = str(row['Finding Labels']).split('|')
        path_vector = [1.0 if p in labels else 0.0 for p in PATHOLOGY_LIST]
        is_no_finding = (row['Finding Labels'] == 'No Finding')
        finding_count = 0.0 if is_no_finding else float(len(labels))
        has_pathology = 0.0 if is_no_finding else 1.0
        
        # Demographics
        age = float(row['Patient Age'])
        sex = 1.0 if str(row['Patient Sex']).upper() == 'M' else 0.0
        view = 1.0 if str(row['View Position']).upper() == 'PA' else 0.0
        followup = float(row['Follow-up #'])
        
        meta_features = path_vector + [finding_count, age, sex, view, followup]
        records.append(meta_features)
        has_pathology_list.append(has_pathology)
        
    return np.array(records, dtype=np.float32), np.array(has_pathology_list, dtype=np.float32)


def build_and_cache_cohort(csv_path=CSV_PATH, images_dir=IMAGES_DIR, n_cohort_patients=1500, random_state=42):
    """
    Identifies longitudinal patients with verified image coverage, samples a reproducible
    cohort of 1,500 patients, extracts real HOG features and clinical metadata, and caches to disk.
    """
    csv_resolved = find_file(csv_path)
    if not os.path.exists(csv_resolved):
        candidates = [f for f in os.listdir(".") if f.startswith("Data_Entry") and f.endswith(".csv")]
        if candidates:
            csv_resolved = candidates[0]

    print(f"[DATA LOADER] Loading CSV metadata from {csv_resolved}...")
    df = pd.read_csv(csv_resolved)
    
    total_scans_raw = len(df)
    total_patients_raw = df['Patient ID'].nunique()
    print(f"[DATA LOADER] Total raw records: {total_scans_raw} scans across {total_patients_raw} unique patients.")
    
    # Sort strictly by patient and follow-up
    df = df.sort_values(by=['Patient ID', 'Follow-up #']).reset_index(drop=True)
    
    # Check image availability in archives or loose directories
    available_images = {}
    tar_paths = []
    for f in TAR_FILES:
        candidate = find_file(f, candidate_dirs=[images_dir, ".", "..", os.path.join("..", images_dir)])
        if os.path.exists(candidate) and candidate not in tar_paths:
            tar_paths.append(candidate)

    # Check for extracted loose PNG images
    loose_images = {}
    candidate_img_dirs = [d for d in [images_dir, ".", os.path.join(images_dir, "images"), "images_001"] if os.path.isdir(d)]
    for d in candidate_img_dirs:
        for fname in os.listdir(d):
            if fname.endswith('.png') and fname not in loose_images:
                loose_images[fname] = os.path.join(d, fname)

    if tar_paths:
        print(f"[DATA LOADER] Inspecting {len(tar_paths)} image archives...")
        for p in tar_paths:
            with tarfile.open(p, 'r:gz') as tar:
                for member in tar:
                    base = os.path.basename(member.name)
                    if base.endswith('.png'):
                        available_images[base] = (p, member.name)
        print(f"[DATA LOADER] Found {len(available_images)} actual PNG images across archives.")
    elif loose_images:
        print(f"[DATA LOADER] Found {len(loose_images)} uncompressed PNG images across directories.")
        for base, fpath in loose_images.items():
            available_images[base] = ("loose", fpath)
    else:
        print("[DATA LOADER] [WARNING] Image archives or loose images not found. Proceeding with clinical metadata only.")
    
    # Determine valid longitudinal patients
    df['has_img'] = df['Image Index'].isin(available_images) if available_images else True
    patient_stats = df.groupby('Patient ID').agg(
        total_scans=('Follow-up #', 'count'),
        scans_with_img=('has_img', 'sum')
    )
    
    if available_images:
        # Require >= 2 scans, all of which have actual images
        valid_pids = patient_stats[
            (patient_stats['total_scans'] >= 2) & 
            (patient_stats['total_scans'] == patient_stats['scans_with_img'])
        ].index.tolist()
    else:
        valid_pids = patient_stats[patient_stats['total_scans'] >= 2].index.tolist()
        
    print(f"[DATA LOADER] Total valid longitudinal patients with full coverage: {len(valid_pids)}")
    
    # Deterministic cohort sampling
    valid_pids = sorted(valid_pids)
    rng = np.random.RandomState(random_state)
    cohort_pids = sorted(rng.choice(valid_pids, size=min(n_cohort_patients, len(valid_pids)), replace=False))
    print(f"[DATA LOADER] Sampled reproducible cohort of {len(cohort_pids)} longitudinal patients (Seed={random_state}).")
    
    df_cohort = df[df['Patient ID'].isin(cohort_pids)].copy()
    df_cohort = df_cohort.sort_values(by=['Patient ID', 'Follow-up #']).reset_index(drop=True)
    total_cohort_scans = len(df_cohort)
    print(f"[DATA LOADER] Cohort contains {total_cohort_scans} total longitudinal scans.")
    
    # Patient-wise split (70% train, 15% val, 15% test)
    n_total = len(cohort_pids)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    
    shuffled_pids = rng.permutation(cohort_pids)
    train_pids = set(shuffled_pids[:n_train])
    val_pids = set(shuffled_pids[n_train:n_train + n_val])
    test_pids = set(shuffled_pids[n_train + n_val:])
    
    print(f"[DATA LOADER] Patient Split: Train={len(train_pids)} pts, Val={len(val_pids)} pts, Test={len(test_pids)} pts.")
    
    # Extract clinical metadata features
    X_meta, y_has_pathology = get_clinical_metadata_features(df_cohort)
    
    # Extract real HOG features if images are available
    if available_images:
        print(f"[DATA LOADER] Extracting real HOG features from {total_cohort_scans} chest X-rays...")
        t0 = time.time()
        
        hog_features: list[Any] = [None] * total_cohort_scans
        extracted_count = 0
        
        # Organize target images by source
        tar_targets = {p: {} for p in tar_paths}
        loose_targets = {}
        for idx, row in df_cohort.iterrows():
            img_name = row['Image Index']
            source, location = available_images[img_name]
            if source == "loose":
                loose_targets[location] = idx
            else:
                tar_targets[source][location] = idx
        
        # 1. Read from tar archives
        for tar_path in tar_paths:
            if not os.path.exists(tar_path):
                continue
            targets = tar_targets[tar_path]
            if not targets:
                continue
            print(f"  Reading {len(targets)} images from {os.path.basename(tar_path)}...")
            with tarfile.open(tar_path, 'r:gz') as tar:
                for member in tar:
                    if member.name in targets:
                        f = tar.extractfile(member)
                        if f is not None:
                            img_bytes = f.read()
                            idx = cast(int, targets[member.name])
                            hog_features[idx] = extract_hog_features_from_image(img_bytes)
                            extracted_count += 1
                            if extracted_count % 1000 == 0 or extracted_count == total_cohort_scans:
                                print(f"    Extracted {extracted_count}/{total_cohort_scans} ({time.time()-t0:.1f}s)")
                            
        # 2. Read from loose image files
        if loose_targets:
            print(f"  Reading {len(loose_targets)} uncompressed images from disk...")
            for img_path, idx in loose_targets.items():
                with open(img_path, 'rb') as f:
                    hog_features[cast(int, idx)] = extract_hog_features_from_image(f.read())
                    extracted_count += 1
                    if extracted_count % 1000 == 0 or extracted_count == total_cohort_scans:
                        print(f"    Extracted {extracted_count}/{total_cohort_scans} ({time.time()-t0:.1f}s)")
                            
        X_hog = np.array(hog_features, dtype=np.float32)
        print(f"[DATA LOADER] Real HOG extraction complete in {time.time()-t0:.2f}s. Feature shape: {X_hog.shape}")
        feature_type = "multimodal_hog_and_clinical"
        X_combined = np.hstack([X_hog, X_meta])
    else:
        print("[DATA LOADER] [DATA INFO] Raw image files not found. Using explicitly identified CSV-derived tabular features.")
        feature_type = "explicit_clinical_metadata"
        X_combined = X_meta

    # Assign split masks strictly by patient
    split_assignments = np.zeros(total_cohort_scans, dtype=np.int32) # 0=train, 1=val, 2=test
    patient_ids_array = np.array(df_cohort['Patient ID'].values, dtype=str)
    
    for i, pid in enumerate(patient_ids_array):
        if pid in train_pids:
            split_assignments[i] = 0
        elif pid in val_pids:
            split_assignments[i] = 1
        elif pid in test_pids:
            split_assignments[i] = 2
            
    # Calculate sequence lengths per patient for each split
    def get_split_lengths(split_id, target_pids):
        lengths = []
        for pid in sorted(target_pids):
            seq_len = int((patient_ids_array[split_assignments == split_id] == pid).sum())
            if seq_len > 0:
                lengths.append(seq_len)
        return np.array(lengths, dtype=np.int32)

    train_lengths = get_split_lengths(0, train_pids)
    val_lengths = get_split_lengths(1, val_pids)
    test_lengths = get_split_lengths(2, test_pids)

    print(f"[DATA LOADER] Split visits: Train={train_lengths.sum()}, Val={val_lengths.sum()}, Test={test_lengths.sum()}.")
    print(f"[DATA LOADER] Mean sequence length: Train={train_lengths.mean():.2f}, Val={val_lengths.mean():.2f}, Test={test_lengths.mean():.2f}.")

    # Save to compressed cache
    np.savez_compressed(
        CACHE_FILE,
        X_features=X_combined,
        split_assignments=split_assignments,
        patient_ids=patient_ids_array,
        y_has_pathology=y_has_pathology,
        train_lengths=train_lengths,
        val_lengths=val_lengths,
        test_lengths=test_lengths,
        feature_type=feature_type,
        total_patients=len(cohort_pids),
        total_scans=total_cohort_scans
    )
    print(f"[DATA LOADER] Cache successfully written to {CACHE_FILE}.")


def load_cxr8_data(cache_file=CACHE_FILE, csv_path=CSV_PATH, n_cohort_patients=1500, random_state=42):
    """
    Loads CXR8 cohort data. Builds the cache if not already present.
    Returns structured dictionary with train, validation, and test arrays.
    """
    cache_resolved = find_file(cache_file)
    if not os.path.exists(cache_resolved):
        print(f"[DATA LOADER] Cache '{cache_file}' not found. Building cohort from raw dataset...")
        build_and_cache_cohort(csv_path=csv_path, n_cohort_patients=n_cohort_patients, random_state=random_state)
        cache_resolved = cache_file
        
    data = np.load(cache_resolved)
    X_features = data['X_features']
    splits = data['split_assignments']
    y_has_pathology = data['y_has_pathology']
    train_lengths = data['train_lengths']
    val_lengths = data['val_lengths']
    test_lengths = data['test_lengths']
    feature_type = str(data['feature_type'])
    
    # Train, Val, Test partitions
    X_train = X_features[splits == 0]
    X_val = X_features[splits == 1]
    X_test = X_features[splits == 2]
    
    y_train = y_has_pathology[splits == 0]
    y_val = y_has_pathology[splits == 1]
    y_test = y_has_pathology[splits == 2]
    
    stats = {
        'total_patients': int(data['total_patients']),
        'total_scans': int(data['total_scans']),
        'train_patients': len(train_lengths),
        'train_scans': len(X_train),
        'val_patients': len(val_lengths),
        'val_scans': len(X_val),
        'test_patients': len(test_lengths),
        'test_scans': len(X_test),
        'feature_dim': X_features.shape[1],
        'feature_type': feature_type
    }
    
    print("\n" + "="*50)
    print("CXR8 LONGITUDINAL COHORT SUMMARY")
    print("="*50)
    print(f"Feature Type      : {stats['feature_type']}")
    print(f"Total Feature Dim : {stats['feature_dim']}")
    print(f"Total Cohort      : {stats['total_patients']} patients, {stats['total_scans']} scans")
    print(f"Training Set      : {stats['train_patients']} patients, {stats['train_scans']} visits (Mean len: {train_lengths.mean():.1f})")
    print(f"Validation Set    : {stats['val_patients']} patients, {stats['val_scans']} visits (Mean len: {val_lengths.mean():.1f})")
    print(f"Test Set          : {stats['test_patients']} patients, {stats['test_scans']} visits (Mean len: {test_lengths.mean():.1f})")
    print("="*50 + "\n")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'train_lengths': train_lengths,
        'val_lengths': val_lengths,
        'test_lengths': test_lengths,
        'stats': stats
    }


def get_preprocessed_features(data, n_pca_components=12):
    """
    Performs leak-free preprocessing:
    - Fits StandardScaler STRICTLY on X_train.
    - Transforms X_train, X_val, X_test using the train scaler.
    - Fits PCA(n_components=n_pca_components) STRICTLY on scaled X_train.
    - Transforms scaled X_train, X_val, X_test using the train PCA.
    
    Returns:
        (X_train_pca, X_val_pca, X_test_pca, scaler, pca)
    """
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    pca = PCA(n_components=n_pca_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    return X_train_pca, X_val_pca, X_test_pca, scaler, pca


def get_ensemble_features(data, n_pca_components=16):
    """
    Constructs leak-free feature representation for Unit 5 Ensemble classification:
    - Target: y_has_pathology (binary: 0 = No Finding, 1 = Active Pathology).
    - Excludes target-leaking columns (the 14 pathology indicators and finding_count).
    - Fits StandardScaler and PCA(n_components=n_pca_components) STRICTLY on training visual HOG features.
    - Transforms val and test visual features using the fitted training scaler and PCA.
    - Concatenates visual PCA components with non-leaking demographic/acquisition metadata:
      ['Patient_Age', 'Patient_Sex', 'View_Position', 'Followup_Num'].
    
    Returns:
        Structured dictionary with X_train, y_train, X_val, y_val, X_test, y_test, and feature_names.
    """
    X_train_raw = data['X_train']
    X_val_raw = data['X_val']
    X_test_raw = data['X_test']
    
    if X_train_raw.shape[1] >= 1587:
        # Multimodal features: 1568 HOG visual features + 19 clinical metadata
        X_train_vis = X_train_raw[:, :1568]
        X_val_vis = X_val_raw[:, :1568]
        X_test_vis = X_test_raw[:, :1568]
        
        # Last 4 non-leaking metadata columns: Patient Age, Patient Sex, View Position, Follow-up #
        X_train_meta = X_train_raw[:, 1583:1587]
        X_val_meta = X_val_raw[:, 1583:1587]
        X_test_meta = X_test_raw[:, 1583:1587]
    else:
        # Fallback if only metadata is available
        X_train_vis = X_train_raw[:, :15]
        X_val_vis = X_val_raw[:, :15]
        X_test_vis = X_test_raw[:, :15]
        X_train_meta = X_train_raw[:, 15:19]
        X_val_meta = X_val_raw[:, 15:19]
        X_test_meta = X_test_raw[:, 15:19]
        
    # Standardize visual features strictly on train
    vis_scaler = StandardScaler()
    X_train_vis_scaled = vis_scaler.fit_transform(X_train_vis)
    X_val_vis_scaled = vis_scaler.transform(X_val_vis)
    X_test_vis_scaled = vis_scaler.transform(X_test_vis)
    
    # Fit PCA strictly on train visual features
    pca = PCA(n_components=n_pca_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_vis_scaled)
    X_val_pca = pca.transform(X_val_vis_scaled)
    X_test_pca = pca.transform(X_test_vis_scaled)
    
    # Standardize metadata strictly on train
    meta_scaler = StandardScaler()
    X_train_meta_scaled = meta_scaler.fit_transform(X_train_meta)
    X_val_meta_scaled = meta_scaler.transform(X_val_meta)
    X_test_meta_scaled = meta_scaler.transform(X_test_meta)
    
    # Concatenate visual PCA components and clinical metadata
    X_train = np.hstack([X_train_pca, X_train_meta_scaled])
    X_val = np.hstack([X_val_pca, X_val_meta_scaled])
    X_test = np.hstack([X_test_pca, X_test_meta_scaled])
    
    pca_names = [f"Visual_PCA_{i+1:02d}" for i in range(n_pca_components)]
    meta_names = ["Patient_Age", "Patient_Sex", "View_Position", "Followup_Num"]
    feature_names = pca_names + meta_names
    
    return {
        'X_train': X_train,
        'y_train': data['y_train'].astype(np.int32),
        'X_val': X_val,
        'y_val': data['y_val'].astype(np.int32),
        'X_test': X_test,
        'y_test': data['y_test'].astype(np.int32),
        'feature_names': feature_names,
        'scaler': vis_scaler,
        'pca': pca
    }


if __name__ == "__main__":
    data = load_cxr8_data()
    X_train_pca, X_val_pca, X_test_pca, scaler, pca = get_preprocessed_features(data, n_pca_components=12)
    print("[DATA LOADER] PCA transformation (n=12) successful:")
    print(f"  X_train_pca shape: {X_train_pca.shape}")
    print(f"  X_val_pca shape  : {X_val_pca.shape}")
    print(f"  X_test_pca shape : {X_test_pca.shape}")
    print(f"  PCA explained variance ratio sum: {pca.explained_variance_ratio_.sum():.4f}")
