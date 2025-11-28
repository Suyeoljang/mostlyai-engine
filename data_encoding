import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import sklearn.model_selection

# ================================================================
# 1. 데이터 로드
# ================================================================
print("=" * 70)
print("데이터 전처리 및 저장")
print("=" * 70)

df = pd.read_csv('cd_trainset.csv')
print(f"\n1. 데이터 로드: {df.shape}")

# 타겟 분리
target_col = 'PROC_EXPOSE_LOG'
y = df[target_col].values
X = df.drop(columns=[target_col])

# Feature 타입 분리
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"   연속형 변수: {len(numerical_cols)}개")
print(f"   범주형 변수: {len(categorical_cols)}개")

# ================================================================
# 2. Train/Val/Test 분할 (고정 seed)
# ================================================================
print("\n2. 데이터 분할")

seed = 42
train_idx, test_idx = sklearn.model_selection.train_test_split(
    np.arange(len(X)), test_size=0.1, random_state=seed
)
train_idx, val_idx = sklearn.model_selection.train_test_split(
    train_idx, test_size=0.1, random_state=seed
)

print(f"   Train: {len(train_idx)} samples")
print(f"   Val:   {len(val_idx)} samples")
print(f"   Test:  {len(test_idx)} samples")

# ================================================================
# 3. 범주형 변수 인코딩 (Train 기준)
# ================================================================
print("\n3. 범주형 변수 인코딩 (Train 기준)")

label_encoders = {}
cat_cardinalities = []

for col in categorical_cols:
    le = LabelEncoder()
    
    # Train set으로만 fit
    X_train_col = X.iloc[train_idx][col].astype(str)
    le.fit(X_train_col)
    
    # 전체 데이터 transform
    def safe_transform(x):
        try:
            return le.transform([str(x)])[0]
        except ValueError:
            return len(le.classes_)  # UNKNOWN
    
    X[col] = X[col].astype(str).apply(safe_transform)
    
    label_encoders[col] = le
    cardinality = len(le.classes_) + 1  # +1 for UNKNOWN
    cat_cardinalities.append(cardinality)
    
    print(f"   {col}: {cardinality}개 카테고리")

# ================================================================
# 4. 저장
# ================================================================
print("\n4. 저장 중...")

save_dir = '/mnt/user-data/outputs/'

# 4-1. 인코딩된 데이터 저장 (CSV)
X['PROC_EXPOSE_LOG'] = y  # 타겟 다시 추가
X.to_csv('encoded_data.csv', index=False)


# 4-2. 인덱스 저장
np.savez('data_split.npz',
         train_idx=train_idx,
         val_idx=val_idx,
         test_idx=test_idx)
print(f"   ✓ 분할 인덱스: {save_dir}data_split.npz")

# 4-3. 메타 정보 저장
metadata = {
    'label_encoders': label_encoders,
    'cat_cardinalities': cat_cardinalities,
    'numerical_cols': numerical_cols,
    'categorical_cols': categorical_cols,
    'target_col': target_col,
    'seed': seed
}

with open('preprocessing_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print(f"   ✓ 메타데이터: {save_dir}preprocessing_metadata.pkl")

# ================================================================
# 5. 완료
# ================================================================
print("\n" + "=" * 70)
print("✅ 전처리 완료!")
print("=" * 70)
print(f"\n저장된 파일:")
print(f"  1. {save_dir}encoded_data.csv          (인코딩된 데이터)")
print(f"  2. {save_dir}data_split.npz            (분할 인덱스)")
print(f"  3. {save_dir}preprocessing_metadata.pkl (메타정보)")
print("=" * 70)
