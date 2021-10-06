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

# +
import sklearn

from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# +
wine = load_wine()
wine_data = wine.data
wine_label = wine.target

print('wine target value:', wine_label)
print('wine target name:', wine.target_names)
# -

wine

print(type(wine_data))

# +
df_wine = pd.DataFrame(wine_data, columns=wine.feature_names)

df_wine['labels'] = wine.target

df_wine.head(3)
# -

df_wine.describe()

df_wine.info()

x_train, x_test, y_train, y_test = train_test_split(wine_data, wine_label,
                                                   test_size=0.2,
                                                   random_state=10000)

dt_clf = DecisionTreeClassifier(random_state=20) #모델 객체 생성
dt_clf.fit(x_train, y_train) #모델 객체 학습
pred = dt_clf.predict(x_test) #테스트 데이터를 이용한 예측

# +
###수업내용 복습
# -

import sklearn

sklearn.__version__

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

iris = load_iris()

# +
iris_data = iris.data
iris_label = iris.target

print('iris target value:', iris_label) #3가지 품종(y)이므로 3가지 결과값 0~2
print('iris target name:', iris.target_names)
# -

# 데이터 구성요소 파악 (data, target, frame, target_name)
iris

# 데이터 타입 확인
print(type(iris_data))

# +
# 데이터 프레임으로 가공
# pandas.DataFrame(data=None, index=Non, columns=None, dtype=None, copy=False)

df_iris = pd.DataFrame(iris_data, columns=iris.feature_names)
# feature_names는 데이터 구성요서 파악 단계 결과값에서 확인 가능

# pandas 열 추가
df_iris['label'] = iris.target

df_iris.head(3)
# -

df_iris.describe()

df_iris.info()

# # iris 데이터셋 분할 (학습/테스트)
#
# Parameter
# - arrays: 분할시킬 데이터 입력
# - test_size: 테스트 데이터셋의 비율(float)이나 갯수(int) (default=0.25)
# - train_size: 학습 데이터셋의 비율이나 개수 (default=None), None을 입력하고 test_size를 지정할 경우 테스트 데이터셋을 뺀 나머지를 훈련 데이터로 사용
# - randome_state: 데이터 분할 시 셔플이 이루어지는데 이를 위한 시드값 (int나 RandomState로 입력)
# - shuffle: 셔플 여부 설정 (default=True)

x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_label,
                                                   test_size=0.2,
                                                   random_state=100)

# Decision Tree 모델 객체 생성
dt_clf = DecisionTreeClassifier(random_state=11)

# Decision Tree 모델 객체 학습
dt_clf.fit(x_train, y_train)

# 테스트 데이터를 이용한 예측
pred = dt_clf.predict(x_test)

# +
# Accuracy(정확도)를 통한 성능 평가
from sklearn.metrics import accuracy_score

print('predict accuracy : {}'.format(accuracy_score(y_test, pred)))
# y_test=정답값, pred=예측값

# +
import sklearn

sklearn.__version__

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

iris = load_iris()

iris_data = iris.data
iris_label = iris.target

print('iris target value:', iris_label) #3가지 품종(y)이므로 3가지 결과값 0~2
print('iris target name:', iris.target_names)

# 데이터 구성요소 파악 (data, target, frame, target_name)
iris

# 데이터 타입 확인
print(type(iris_data))

# 데이터 프레임으로 가공
# pandas.DataFrame(data=None, index=Non, columns=None, dtype=None, copy=False)

df_iris = pd.DataFrame(iris_data, columns=iris.feature_names)
# feature_names는 데이터 구성요서 파악 단계 결과값에서 확인 가능

# pandas 열 추가
df_iris['label'] = iris.target

df_iris.head(3)

df_iris.describe()

df_iris.info()

# iris 데이터셋 분할 (학습/테스트)

Parameter
- arrays: 분할시킬 데이터 입력
- test_size: 테스트 데이터셋의 비율(float)이나 갯수(int) (default=0.25)
- train_size: 학습 데이터셋의 비율이나 개수 (default=None), None을 입력하고 test_size를 지정할 경우 테스트 데이터셋을 뺀 나머지를 훈련 데이터로 사용
- randome_state: 데이터 분할 시 셔플이 이루어지는데 이를 위한 시드값 (int나 RandomState로 입력)
- shuffle: 셔플 여부 설정 (default=True)

x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_label,
                                                   test_size=0.2,
                                                   random_state=100)

# Decision Tree 모델 객체 생성
dt_clf = DecisionTreeClassifier(random_state=11)

# Decision Tree 모델 객체 학습
dt_clf.fit(x_train, y_train)

# 테스트 데이터를 이용한 예측
pred = dt_clf.predict(x_test)

# Accuracy(정확도)를 통한 성능 평가
from sklearn.metrics import accuracy_score

print('predict accuracy : {}'.format(accuracy_score(y_test, pred)))
# y_test=정답값, pred=예측값
