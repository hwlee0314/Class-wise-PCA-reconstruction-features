# PCA 재구성오차를 이용한 피쳐엔지니어링 기법

## 개요
**PCA 재구성오차 기법**은 전체 데이터를 PCA룰 통해 차원을 축소한 후 다시 원래 차원으로 복원했을 때 발생하는 **잔차를 새로운 피쳐로 활용**하는 기법

##  사용하는 이유
- **도메인 지식이 없을 때** 간단하게 유의미한 수많은 변수들을 생성항 수 있음
- 원본 데이터에서 놓칠 수 있는 **숨겨진 패턴**을 발견
- 클래스별로 서로 다른 재구성 오차 패턴을 생성하여 **성능 향상**

## 구현 과정

### 1. 데이터 전처리
```python
# X로 시작하는 피쳐 컬럼들만 선택
cols = [col for col in train.columns if col.startswith("X_")]

# StandardScaler로 정규화
scaler = StandardScaler()
train[cols] = scaler.fit_transform(train[cols])
test[cols] = scaler.transform(test[cols])
```

### 2. 클래스별 PCA 적용
```python
for i in range(21):  # 21개 클래스에 대해 반복
    mask = train["target"] == i  # 각 클래스별로 마스킹
    
    # PCA 모델 생성 (분산의 90% 보존)
    decomposition = PCA(n_components=0.9, svd_solver='full', random_state=42)
    decomposition.fit(train.loc[mask,cols])  # 해당 클래스 데이터로만 학습
```

### 3. 재구성오차 계산
```python
# 차원 축소 → 복원 → 오차 계산
x = decomposition.transform(train[cols])      # 차원 축소
x = decomposition.inverse_transform(x)        # 원래 차원으로 복원
residual = train[cols] - x                    # 재구성 오차 계산
```

## 4. 사용법

1. 클래스 개수에 따라 반복문 범위를 조정합니다 (예: 이진분류면 `range(2)`)
2. `n_components` 파라미터로 보존할 분산의 비율을 결정합니다.
3. 다만 불필요한 변수들도 많이 생성되므로 피쳐 셀렉션 등 중요도 기준으로 걸러주는게 좋을듯