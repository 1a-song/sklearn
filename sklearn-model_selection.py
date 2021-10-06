# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] id="gw7UTxVjNjep"
# # 데이터 셋 분할

# + executionInfo={"elapsed": 1598, "status": "ok", "timestamp": 1633517312640, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="73LqVpm6jIpx"
# 라이브러리 불러오기
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 259, "status": "ok", "timestamp": 1633517317484, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="T3tz4jq5uOnO" outputId="1df0aff9-7e9f-4edd-c87b-c08ee7f4b259"
iris = load_iris()
dt_clf = DecisionTreeClassifier()

train_data = iris.data
train_label = iris.target

dt_clf.fit(train_data, train_label)
pred = dt_clf.predict(train_data)

print('prediction accuracy:', accuracy_score(train_label, pred))

# + [markdown] id="TwxU6BvOReKu"
# # train_test_split() API - 데이터 셋의 분할(학습/ 테스트)을 쉽게 할 수 있는 API
#
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#
# • shuffle: 데이터셋 분리 전 미리 섞을지 여부 (default=True)
#
# • random_state: 실행시 마다 동일한 난수를 생성하기 위한 값
#
# • 튜플 형태로 분할(학습/ 테스트)된 데이터 리턴

# + executionInfo={"elapsed": 265, "status": "ok", "timestamp": 1633517329262, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="0G9trkFDjWmZ"
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# + executionInfo={"elapsed": 273, "status": "ok", "timestamp": 1633517349566, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="BWV3KvYMTz0D"
iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=100)

# + [markdown] id="WIKiJE6TUOg3"
# * 분할된 데이터 중 학습데이터를 이용해 학습
# * 분할된 데이터 중 테스트 데이터를 이용해 성능 검증
#

# + executionInfo={"elapsed": 247, "status": "ok", "timestamp": 1633517354399, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="cA6eMiWIUUJM"
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# + executionInfo={"elapsed": 251, "status": "ok", "timestamp": 1633517356214, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="VPtwqgsGUXez"
# Decision Tree 모델 객체 생성
dt_clf = DecisionTreeClassifier()

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 268, "status": "ok", "timestamp": 1633517359152, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="p2fL95s4VDBZ" outputId="fc884e4f-3666-4904-d87a-c306114ca408"
# Decision Tree 모델 객체 학습
dt_clf.fit(x_train, y_train)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 280, "status": "ok", "timestamp": 1633517362886, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="ikEjEQA2VC7P" outputId="bc2cbcf4-83b3-49b0-b238-71272f1a38d3"
# 테스트 데이터를 이용한 예측
pred = dt_clf.predict(x_test)

print('prediction accuracy : {}'.format(accuracy_score(y_test, pred)))

# + [markdown] id="iT0rCtWyVbTp"
# ## KFold API - K겹 교차 검증을 쉽게 적용 할 수 있는 Cross Validation API

# + executionInfo={"elapsed": 270, "status": "ok", "timestamp": 1633517368754, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="3eWZ6p9KVPpA"
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
import numpy as np

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 267, "status": "ok", "timestamp": 1633517436923, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="Bs3obTBXWn7b" outputId="8255ca3e-d5dd-4dd9-cde2-e9e4b16e038e"
# Iris 데이터셋 로드 및 KFold 객체 생성
iris = load_iris()
features = iris.data
label = iris.target
kfold = KFold(n_splits=5)

print('iris Data set Size:', features.shape)

# + executionInfo={"elapsed": 386, "status": "ok", "timestamp": 1633517585540, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="JXUgSm8Wf_t6"
# Decision Tree 객체 생성 및 Fold 별 지표 값 저장을 위한 리스트 생성
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

dt_clf = DecisionTreeClassifier(random_state=20)
cv_accuracy = []

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 246, "status": "ok", "timestamp": 1633517587201, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="wphhWaJbgP-K" outputId="836a555c-9a92-40fa-cb6d-deaadb1c7ccb"
fold_index = 0

for train_index, test_index in kfold.split(features):
    # KFold 객체의 split() 함수를 이용해 분할된 Fold 별 인덱스 생성
    # 생성된 인덱스를 이용해 데이터셋 분할
    x_train, x_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    print(train_index)
    print(test_index)

    # 분할된 Fold를 이용하여 모델 학습 
    dt_clf.fit(x_train, y_train)

    # 성능 측정
    pred = dt_clf.predict(x_test)

    fold_index += 1

    # Fold 별 정확도 지표 측정 및 cv_accuracy 리스트에 추가
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]

    print("\n #{0} fold accuracy : {1}, train size : {2}, val size : {3}".format(fold_index, accuracy, train_size, test_size))
    print("#{0} val index : {1}".format(fold_index, test_index))
    
    cv_accuracy.append(accuracy)

    # cv_accuracy 리스트에 저장된 Fold 별 정확도 값을 이용해 평균 정확도 계산
    # 계산된 KFold 평가지표(평균 정확도) 출력
print('\n ## avg val accuracy :', np.mean(cv_accuracy))

# + [markdown] id="F6J4haDRmWAw"
# # Stratified K Fold 
# * Imbalanced(불균형) 분포를  가진 데이터  셋을 위한  방식 (KFold와의 차이점이기도 함)
# * Stratified KFold는 데이터 셋의 비율을 반영하여 샘플링

# + executionInfo={"elapsed": 268, "status": "ok", "timestamp": 1633518147030, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="Q4eEAagnm4dG"
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

# + colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 252, "status": "ok", "timestamp": 1633518397882, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="WPdgCLvfm57u" outputId="2dc15dc3-dcba-4eeb-9cb2-981d982a1320"
# Iris 데이터셋 로드하여 DataFrame 객체 생성
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

df.tail()

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 382, "status": "ok", "timestamp": 1633518406398, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="y1J9Va2ZSLSK" outputId="931a9b6c-f0be-4f7f-94f6-8e83bf376800"
df.info()

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 248, "status": "ok", "timestamp": 1633526931174, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="99miCOy9nEB4" outputId="cc8d3346-89c0-4fcb-96c6-6641892afb46"
# n_splits 인자를 3으로 지정하여 KFold 객체 생성
# KFold API로 생성한 인덱스로 데이터셋 분할
from sklearn.datasets import load_iris

features = iris.data
label = iris.target
kfold = KFold(n_splits=3)

# DataFrame 객체의 value_counts() 메서드를 이용해 데이터 분포 확인

df = pd.DataFrame(features, columns=iris.feature_names)
df['target'] = label

fold_index = 0
for train_index, test_index in kfold.split(features):
  feature_index = df['target'].iloc[train_index]
  label_index = df['target'].iloc[test_index] 
  print("### Cross validation : {0}".format(fold_index))
  print("Train label distribution : \n {0}".format(feature_index.value_counts()))
  print("Test label distribution : \n {0}".format(label_index.value_counts()))
  print()
  fold_index += 1

# + [markdown] id="fqIs35RpoywD"
#
# ### StratifiedKFold API를 이용하여 Fold 생성 하여 데이터의 분포 확인

# + executionInfo={"elapsed": 276, "status": "ok", "timestamp": 1633525867982, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="zckFwLzknTlB"
# n_splits 인자를 3으로 지정하여 StartifiedKFold 객체 생성
from sklearn.datasets import load_iris
import sklearn

skfold = sklearn.model_selection.StratifiedKFold(n_splits=3)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1633527300783, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="BkuNa3AipJ0A" outputId="f3ef9eff-f18c-45c1-c61d-5d1e0f508bc5"
# StartifiedKFold 객체의 split() 메소드에  분포 확인을 위한 label 데이터셋을 인자로 전달
# StartifiedKFold API로 생성한 인덱스로 데이터셋 분할

iris = load_iris()
features = iris.data
label = iris.target

df = pd.DataFrame(data=features, columns=iris.feature_names)
df['target'] = label


  # DataFrame 객체의 value_counts() 메서드를 이용해 데이터 분포 확인


fold_index = 0

for train_index, test_index in skfold.split(df, df['target']):
  feature_index = df['target'].iloc[train_index]
  label_index = df['target'].iloc[test_index]
  
  print("### Cross validation = {0}".format(fold_index))
  print("Train label distribution : \n {0}".format(feature_index.value_counts()))
  print("Val label distribution : \n {0}".format(label_index.value_counts()))
  print()

  fold_index += 1




# + [markdown] id="difin4K7qhyA"
# ## StratifiedKFold API를 이용한 Cross Validation

# + executionInfo={"elapsed": 252, "status": "ok", "timestamp": 1633528106281, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="Z3hvH5kIppa1"
# Iris 데이터셋 load
# DecisionTreeClassifier 모델 객체 생성

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import sklearn

dt_clf = DecisionTreeClassifier(random_state=120)

# + executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1633528107277, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="7CN5cghBqomJ"
iris = load_iris()
features = iris.data
label = iris.target

# df = pd.DataFrame(data=features, columns=iris.feature_names)
# df['target'] = label

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 262, "status": "ok", "timestamp": 1633528178496, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="tzIfHjl1qru3" outputId="462353dd-9e0c-4975-bfed-426f72e58ebd"
# Fold 별 지표 값 저장을 위한 lst_accuracy 리스트 생성
# 분할된 데이터 Fold로 학습 및 예측 수행

skfold = sklearn.model_selection.StratifiedKFold(n_splits=3)

lst_accuracy = []
fold_index = 0

for train_index, test_index in skfold.split(features, label):
  
  # StartifiedKFold API로 생성한 인덱스로 데이터셋 분할
  x_train, x_test = features[train_index], features[test_index]
  y_train, y_test = label[train_index], label[test_index]
  print(train_index)
  print(test_index)

  # 분할된 데이터를 이용해 모델 학습 및 성능 테스트
  dt_clf.fit(x_train, y_train)

  pred = dt_clf.predict(x_test)

  fold_index += 1

  # Fold 별 정확도 지표 측정 및 lst_accuracy 리스트에 추가
  accuracy = np.round(accuracy_score(y_test, pred), 4)
  train_size = x_train.shape[0]
  test_size = x_test.shape[0]

  print("\n #{0} fold accuracy : {1}, Train size : {2}, val size : {3}".format(fold_index, accuracy, train_size, test_size))
  print("###{0} val index : {1}".format(fold_index, test_index))

  lst_accuracy.append(accuracy)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 245, "status": "ok", "timestamp": 1633528188850, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="Ta4QFeQVrNIq" outputId="aae376ea-7fc7-474d-feb9-8831abb8de1f"
# lst_accuracy 리스트에 저장된 Fold 별 정확도  값을 이용해 평균 정확도 계산
# 계산된 KFold 평가 지표(평균 정확도) 출력

# 평균 정확도
print('\n ## avg val accuracy :', np.mean(lst_accuracy))

# + [markdown] id="4hGn1SM5vvVw"
# ## cross_val_score() API
#
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
#
# Argument
# * estimator : 구현하려는 모델 (Classification, Regression, ...)
#
# * x : Dataset
# * y : Label dataset
# * scoring : 성능 평가 지표
# * cv : cross validation의 fold index
#
# Return : list 형태의 fold별 성능지표 (검증결과)

# + [markdown] id="rT0BVeM7v0a-"
# ## cross_val_score() API를 이용하여 교차검증 성능 지표 계산

# + executionInfo={"elapsed": 247, "status": "ok", "timestamp": 1633528194718, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="H0G2xohcv9W5"
# Iris 데이터셋 load
# DecisionTreeClassifier 모델 객체 생성
from sklearn.tree import DecisionTreeClassifier 
from sklearn.datasets import load_iris

iris = load_iris() 
data = iris.data
label = iris.target

dt_clf = DecisionTreeClassifier(random_state=156)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 253, "status": "ok", "timestamp": 1633528197823, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="Dg3JAtPBwW-Z" outputId="e47795d5-8448-4499-ff80-d9e71aa007ad"
# cross_val_score() API를 이용하여 교차검증 성능 지표를 list 형태로 생성
# score 리스트에 저장된 Fold 별 정확도 값을 이용해 평균 정확도 계산
# 계산된 KFold 평가지표(평균 정확도) 출력
# cross_val_score() API를 이용하여 교차검증 성능 지표를 list 형태로 생성
# score 리스트에 저장된 Fold 별 정확도 값을 이용해 평균 정확도 계산
# 계산된 KFold 평가지표(평균 정확도) 출력
from sklearn.model_selection import cross_val_score
import numpy as np

scores = cross_val_score(estimator = dt_clf,
                         X=data, y=label,
                         scoring = 'accuracy',
                         cv = 3)

print('Fold val accuracy :', np.round(scores,4))
print('Avg val accuracy :', np.round(np.mean(scores), 4))

# + [markdown] id="KOtUKEtAM31z"
# # GridSearchCV API
#
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
#
# 모델의 최적 하이퍼 파라미터를 찾아주는 API
# * param_grid argument로 전달된 파라미터를 순차적으로 적용해 학습 및 테스트
#
# Argument
# * estimator
# * param_grid : 모델 튜닝에 사용될 파라미터 정보 (dict)
# * scoring
# * cv 
# * refit : 최적의 파라미터를 모델에 적용해 재학습 시킬지 여부 (default=True)

# + [markdown] id="CWi36qYqNA2v"
# ## GridSearchCV API를 이용하여 최적의 모델 학습 시키기

# + executionInfo={"elapsed": 246, "status": "ok", "timestamp": 1633528202233, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="XKt-v-Y1xAxg"
# Iris 데이터 셋 load
# 학습/ 테스트 데이터 셋으로 분할
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 

iris = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=121)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 256, "status": "ok", "timestamp": 1633528209775, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="OFOx2NzqM-ll" outputId="daf826ef-d6ef-4ec5-81a9-838fc049f253"
# GridSearchCV API 사용한 하이퍼파라미터 조정
# 관심있는 매개변수들을 대상으로 가능한 모든 조합을 시도해보는 것!

# 검증을 진행할 모델의 파라미터 정보 지정
# 모델 객체와 파라미터 정보를 인자로 전달하여 GridSearchCV 객체 생성
# GridSearchCV 객체의 fit() 메서드를 이용해 학습 및 검증 진행
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

dtree = DecisionTreeClassifier()
params = {'max_depth':[1,2,3], 'min_samples_split':[2,3]}

grid_dtree = GridSearchCV(dtree, param_grid=params, cv=3, refit=True)
grid_dtree.fit(x_train, y_train)

# + colab={"base_uri": "https://localhost:8080/", "height": 238} executionInfo={"elapsed": 273, "status": "ok", "timestamp": 1633528212428, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="nY1jZSkuNZaD" outputId="23ee51cf-8d34-4213-a31e-7b8173b6c8f9"
# GridSearchCV 객체를 이용한 학습 및 검증 결과 확인
import pandas as pd

scores_df = pd.DataFrame(grid_dtree.cv_results_)
scores_df[['params', 'mean_test_score','rank_test_score','split0_test_score',
           'split1_test_score', 'split2_test_score']]

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 252, "status": "ok", "timestamp": 1633528214471, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="qtfNO3jKNiJn" outputId="700a2005-f48b-44e5-b2e9-62e2ee8b2b16"
# 최적의 파라미터 정보 확인
print('Optimal parameter :', grid_dtree.best_params_)
print('Max accuracy : {0: .4f}'.format(grid_dtree.best_score_))

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 259, "status": "ok", "timestamp": 1633528215908, "user": {"displayName": "\uc1a1\uc6d0\uc544\uc815\ubcf4\ud1b5\uc2e0\ub300\ud559\uc6d0", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "16711378382455460525"}, "user_tz": -540} id="n3esthnWNos3" outputId="fcc778cc-d125-4573-9b9b-05cc4b84d819"
# 최적의 파라미터로 학습된 모델의 성능 검증

from sklearn.metrics import accuracy_score

estimator = grid_dtree.best_estimator_
pred = estimator.predict(x_test)

print('Test dataset accuracy : {0: .4f}'.format(accuracy_score(y_test, pred)))
