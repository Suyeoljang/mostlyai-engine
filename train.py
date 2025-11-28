
import math
import random
from typing import Literal, NamedTuple, Optional
from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.special
import sklearn.metrics
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.optim
from torch import Tensor
import pickle

# TabM 관련 임포트
try:
    import tabm
    import rtdl_num_embeddings
    TABM_AVAILABLE = True
except ImportError:
    TABM_AVAILABLE = False
    print("⚠️  tabm 패키지가 설치되지 않았습니다.")
    exit(1)

# ================================================================
# 설정
# ================================================================
print("=" * 70)
print("TabM 빠른 학습 - 전처리된 데이터 사용")
print("=" * 70)

# 시드 고정
seed = 42
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n사용 디바이스: {device}")

# ================================================================
# 1. 전처리된 데이터 로드 (빠름!)
# ================================================================
print("\n" + "=" * 70)
print("1. 전처리된 데이터 로드")
print("=" * 70)

data_dir = '/mnt/user-data/outputs/'

# 1-1. 인코딩된 데이터
df = pd.read_csv('encoded_data.csv')
print(f"✓ 인코딩된 데이터 로드: {df.shape}")

# 1-2. 분할 인덱스
split_data = np.load('data_split.npz')
train_idx = split_data['train_idx']
val_idx = split_data['val_idx']
test_idx = split_data['test_idx']
print(f"✓ 분할 인덱스 로드")
print(f"  Train: {len(train_idx)} samples")
print(f"  Val:   {len(val_idx)} samples")
print(f"  Test:  {len(test_idx)} samples")

# 1-3. 메타 정보
with open('preprocessing_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

label_encoders = metadata['label_encoders']
cat_cardinalities = metadata['cat_cardinalities']
numerical_cols = metadata['numerical_cols']
categorical_cols = metadata['categorical_cols']
target_col = metadata['target_col']

print(f"✓ 메타정보 로드")
print(f"  연속형: {len(numerical_cols)}개")
print(f"  범주형: {len(categorical_cols)}개")
print(f"  카디널리티: {cat_cardinalities}")

# 타겟 분리
y = df[target_col].values
X = df.drop(columns=[target_col])

# ================================================================
# 2. numpy 배열 변환
# ================================================================
print("\n" + "=" * 70)
print("2. numpy 배열 변환")
print("=" * 70)

X_num = X[numerical_cols].values
X_cat = X[categorical_cols].values
Y_numpy = y

data_numpy = {
    'train': {'x_num': X_num[train_idx].astype(np.float32),
              'x_cat': X_cat[train_idx].astype(np.int64),
              'y': Y_numpy[train_idx].astype(np.float32)},
    'val': {'x_num': X_num[val_idx].astype(np.float32),
            'x_cat': X_cat[val_idx].astype(np.int64),
            'y': Y_numpy[val_idx].astype(np.float32)},
    'test': {'x_num': X_num[test_idx].astype(np.float32),
             'x_cat': X_cat[test_idx].astype(np.int64),
             'y': Y_numpy[test_idx].astype(np.float32)}
}

print("✓ 변환 완료")

# ================================================================
# 3. 데이터 전처리
# ================================================================
print("\n" + "=" * 70)
print("3. 데이터 전처리")
print("=" * 70)

# 연속형 변수 정규화
x_num_train = data_numpy['train']['x_num']
noise = np.random.default_rng(0).normal(0.0, 1e-5, x_num_train.shape).astype(x_num_train.dtype)

preprocessing = sklearn.preprocessing.QuantileTransformer(
    n_quantiles=max(min(len(train_idx) // 30, 1000), 10),
    output_distribution='normal',
    subsample=10**9,
).fit(x_num_train + noise)

for part in data_numpy:
    data_numpy[part]['x_num'] = preprocessing.transform(data_numpy[part]['x_num'])

print("✓ 연속형 변수 QuantileTransformer 적용")

# 타겟 표준화
class RegressionLabelStats(NamedTuple):
    mean: float
    std: float

task_type: Literal['regression', 'binclass', 'multiclass'] = 'regression'
n_classes = None

Y_train = data_numpy['train']['y'].copy()
regression_label_stats = RegressionLabelStats(
    Y_train.mean().item(), Y_train.std().item()
)
Y_train = (Y_train - regression_label_stats.mean) / regression_label_stats.std
data_numpy['train']['y'] = Y_train

print(f"✓ 타겟 표준화 (mean={regression_label_stats.mean:.4f}, std={regression_label_stats.std:.4f})")

# ================================================================
# 4. PyTorch 텐서 변환
# ================================================================
print("\n" + "=" * 70)
print("4. PyTorch 텐서 변환")
print("=" * 70)

data = {
    part: {key: torch.tensor(value, device=device) 
           for key, value in part_data.items()}
    for part, part_data in data_numpy.items()
}

print("✓ 텐서 변환 완료")

# ================================================================
# 5. TabM 모델 생성
# ================================================================
print("\n" + "=" * 70)
print("5. TabM 모델 생성")
print("=" * 70)

n_num_features = len(numerical_cols)
n_cat_features = len(categorical_cols) if len(categorical_cols) > 0 else None

num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
    rtdl_num_embeddings.compute_bins(
        torch.tensor(data_numpy['train']['x_num'], device='cpu'),
        n_bins=48
    ),
    d_embedding=32, #16
    activation=False,
    version='B',
)

model = tabm.TabM.make(
    n_num_features=n_num_features,
    cat_cardinalities=cat_cardinalities if n_cat_features else None,
    d_out=1 if n_classes is None else n_classes,
    num_embeddings=num_embeddings,
).to(device)

print(f"✓ 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
print(f"✓ 앙상블 크기 (k): {model.backbone.k}")

# ================================================================
# 6. 학습 설정
# ================================================================
print("\n" + "=" * 70)
print("6. 학습 설정")
print("=" * 70)

# 하이퍼파라미터 
LEARNING_RATE = 1e-3 
WEIGHT_DECAY = 1e-3   
BATCH_SIZE = 256
N_EPOCHS = 1000
PATIENCE = 20

optimizer = torch.optim.AdamW(model.parameters(), 
                               lr=LEARNING_RATE, 
                               weight_decay=WEIGHT_DECAY)
gradient_clipping_norm: Optional[float] = 1.0
share_training_batches = True

# Learning Rate Scheduler 추가!
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='max', 
                              factor=0.5, patience=10, 
                              min_lr=1e-6)

# Eval 배치 크기 자동 설정
if device.type == 'cuda':
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_memory_gb >= 40:
        eval_batch_size = 65536
    elif gpu_memory_gb >= 20:
        eval_batch_size = 32768
    elif gpu_memory_gb >= 12:
        eval_batch_size = 16384
    else:
        eval_batch_size = 8096
else:
    eval_batch_size = 4096

print(f"✓ Learning rate: {LEARNING_RATE}")
print(f"✓ Weight decay: {WEIGHT_DECAY}")
print(f"✓ Batch size: {BATCH_SIZE}")
print(f"✓ Eval batch size: {eval_batch_size}")
print(f"✓ Max epochs: {N_EPOCHS}")
print(f"✓ Patience: {PATIENCE}")
print(f"✓ LR Scheduler: ReduceLROnPlateau")

# ================================================================
# 7. 학습 함수
# ================================================================

def apply_model(part: str, idx: Tensor) -> Tensor:
    x_num = data[part]['x_num'][idx]
    x_cat = data[part]['x_cat'][idx] if n_cat_features is not None else None
    return model(x_num, x_cat)


def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
    if task_type == 'regression':
        base_loss_fn = nn.functional.mse_loss
    else:
        base_loss_fn = nn.functional.cross_entropy
    
    y_pred = y_pred.flatten(0, 1).squeeze(-1)
    
    if share_training_batches:
        y_true = y_true.repeat_interleave(model.backbone.k)
    else:
        y_true = y_true.flatten(0, 1)
    
    return base_loss_fn(y_pred, y_true)


@torch.no_grad()
def evaluate(part: str) -> float:
    model.eval()
    
    y_pred: np.ndarray = (
        torch.cat([
            apply_model(part, idx)
            for idx in torch.arange(len(data[part]['y']), device=device).split(eval_batch_size)
        ])
        .cpu()
        .numpy()
    )
    
    if task_type == 'regression':
        y_pred = y_pred * regression_label_stats.std + regression_label_stats.mean
    
    if task_type != 'regression':
        y_pred = scipy.special.softmax(y_pred, axis=-1)
    y_pred = y_pred.mean(1)
    
    y_true = data[part]['y'].cpu().numpy()
    
    score = (
        -(sklearn.metrics.root_mean_squared_error(y_true, y_pred) ** 0.5)
        if task_type == 'regression'
        else sklearn.metrics.accuracy_score(y_true, y_pred.argmax(1))
    )
    
    return float(score)

# ================================================================
# 8. 학습 시작
# ================================================================
print("\n" + "=" * 70)
print("7. 학습 시작")
print("=" * 70)

print(f'\n학습 전 Test RMSE: {-evaluate("test"):.4f}')

best_val_score = -np.inf
best_epoch = -1
best_state = None
no_improvement_count = 0

for epoch in range(N_EPOCHS):
    model.train()
    
    epoch_losses = []
    for batch_idx in torch.randperm(len(data['train']['y']), device=device).split(BATCH_SIZE):
        optimizer.zero_grad()
        
        y_pred = apply_model('train', batch_idx)
        
        if share_training_batches:
            y_true = data['train']['y'][batch_idx]
        else:
            batch_idx_k = torch.stack([
                torch.randperm(len(data['train']['y']), device=device)[:len(batch_idx)]
                for _ in range(model.backbone.k)
            ], dim=1)
            y_true = data['train']['y'][batch_idx_k]
        
        loss = loss_fn(y_pred, y_true)
        epoch_losses.append(loss.item())
        
        loss.backward()
        
        if gradient_clipping_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_norm)
        
        optimizer.step()
    
    avg_loss = np.mean(epoch_losses)
    val_score = evaluate('val')
    test_score = evaluate('test')
    
    # LR Scheduler 업데이트
    scheduler.step(val_score)
    current_lr = optimizer.param_groups[0]['lr']
    
    if val_score > best_val_score:
        best_val_score = val_score
        best_epoch = epoch
        best_state = deepcopy(model.state_dict())
        no_improvement_count = 0
        print(f'* [epoch] {epoch:<3} [loss] {avg_loss:.4f} [val] {val_score:.4f} [test] {test_score:.4f} [lr] {current_lr:.6f}')
    else:
        no_improvement_count += 1
        if epoch % 5 == 0:
            print(f'  [epoch] {epoch:<3} [loss] {avg_loss:.4f} [val] {val_score:.4f} [test] {test_score:.4f} [lr] {current_lr:.6f}')
    
    if no_improvement_count >= PATIENCE:
        print(f'\nEarly stopping at epoch {epoch}')
        break

# ================================================================
# 9. 최종 평가
# ================================================================
print("\n" + "=" * 70)
print("8. 최종 평가")
print("=" * 70)

model.load_state_dict(best_state)

final_val_score = evaluate('val')
final_test_score = evaluate('test')

print(f'\nBest epoch: {best_epoch}')
print(f'Final Validation RMSE: {-final_val_score:.4f}')
print(f'Final Test RMSE: {-final_test_score:.4f}')

# ================================================================
# 10. 모델 저장
# ================================================================
print("\n" + "=" * 70)
print("9. 모델 저장")
print("=" * 70)

save_path = 'tabm_cd_model.pt'
torch.save({
    'model_state_dict': best_state,
    'regression_label_stats': regression_label_stats,
    'preprocessing': preprocessing,
    'label_encoders': label_encoders,
    'numerical_cols': numerical_cols,
    'categorical_cols': categorical_cols,
    'cat_cardinalities': cat_cardinalities,
    'best_epoch': best_epoch,
    'best_val_score': best_val_score,
    'config': {
        'lr': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'batch_size': BATCH_SIZE,
    }
}, save_path)

print(f"✓ 모델 저장: {save_path}")

print("\n" + "=" * 70)
print("학습 완료!")
print("=" * 70)
print(f"\n최종 Test RMSE: {-final_test_score:.4f}")
print(f"Best epoch: {best_epoch}")
print("=" * 70)
