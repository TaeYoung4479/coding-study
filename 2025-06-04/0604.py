import pandas as pd

df = pd.read_csv('C:/Users/user/codingstudy/data/Titanic.csv')
print(df.head())

#결측값 처리 및 그룹핑
#isnull()함수를 통해 결측값 확인
print(df.isnull())
print(df.isnull().sum())

#notnull()함수를 통한 null값 제외
df_notnull_cabin = df[df['Cabin'].notnull()]
df.shape
df_notnull_cabin.shape

#dropna()함수를 통해 결측값을 가진 행이나 열 제거
df_dropna_age = df['Age'].dropna()
df_dropna_age.shape

#결측치 대체
#fiina()함수를 통해 대체
df_copy = df.copy()
avg_age = df['Age'].mean()
df_copy['Age'] = df['Age'].fillna(avg_age)
df_copy.isnull().sum()
df_copy.loc[:, ['Age']]

#특정 값에 기반 요약하는 groupby()함수 사용
df_1 = df.groupby('Pclass').mean()
#TypeError: agg function failed [how->mean,dtype->object
#해당 오류 발생
df.info()
#각 컬럼에서 object형식의 컬럼 집계 불가능

#실행을 위해 int, float형식의 컬럼들로 새로운 데이터프레임 생성
#[] -> 하나의 열 선택(Series)
#[[]] -> DataFrame, 해당 컬럼을 데이터프레임으로 구조 유지
new_col = ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
df_new = df[new_col]
df_new

df_new_1 = pd.DataFrame(df[['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])
df_new_1

#PassengerId컬럼 기준으로 각 컬럼 평균 집계
df_group = df_new.groupby('PassengerId').mean()
df_group

#pclass기준 Age 평균 나이 구하기
df_group_class = df_new.groupby('Pclass').mean()
df_group_class[['Age']]

#agg()함수를 통해 각 컬럼별 다른 집계함수 적용 가능
df_group_agg = df_new.groupby('Pclass').agg({'Age' : 'mean', 'Fare' : 'count'})
df_group_agg.head()

#Numpy 배열
#list구조를 활용한 배열구조
#--추후 공부

#데이터 셋 정의
#iris Dataset
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
iris.data
print(iris.data.shape)
iris.target
print(type(iris))
print(iris.DESCR)
iris.target_names
iris.keys()
iris.values()
iris.feature_names
iris['target'].shape

#iris데이터 셋 데이터프레임화
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_iris.tail()
df_iris['target'] = iris.target
df_iris