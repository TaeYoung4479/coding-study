import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('C:/Users/user/codingstudy/data/Customer_Data.csv')
print(df.head())

#standardscaler를 활용한 정규화
scaler = StandardScaler()

#standardscaler는 2차원 형식으로 데이터를 받아야함 -> df[['age']]
df['age'] = scaler.fit_transform(df[['age']])
df['age']


##최빈값
print(df['income'].mode())

#박스-콕스 변환 (양수만 사용 가능)
#로그 변환이나 제곱근 변환의 일반화
from sklearn.preprocessing import PowerTransformer

#method -> 디폴트 -> 여-존슨 변환
transform = PowerTransformer(method = 'box-cox')
df['box'] = transform.fit_transform(df[['age']])
df['box']

#여-존슨 변환(음수, 양수 모두 처리 가능)
transform = PowerTransformer()
df['yeo'] = transform.fit_transform(df[['age']])
df['yeo']

#min-max
df = pd.read_csv('C:/Users/user/codingstudy/data/Customer_Data.csv')
print(df.head())
df.isnull().sum()

#특정 컬럼 min-max변환 후 상위 5, 하위 5% 구하기
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df['MinMax'] = scaler.fit_transform(df[['income']])
df['MinMax']

print(df['MinMax'].quantile(0.05))
print(df['MinMax'].quantile(0.95))

#상위 하위 데이터 조회, 계산
df = pd.read_csv('C:/Users/user/codingstudy/data/covid-vaccination-vs-death_ratio.csv')
df.head()
df1 = df.groupby('country').max()
df1 = df1.sort_values('ratio', ascending = False)
#100%넘는 나라 제거
df2 = df1[df1['ratio'] <= 100]
df2.sort_values('ratio', ascending = False)
df2
top = df2['ratio'].head(10).mean()
top
bottom = df2['ratio'].tail(10).mean()
bottom

#상관관계
import pandas as pd
df = pd.read_csv('C:/Users/user/codingstudy/data/winequality-red.csv')
df

df_corr = df.corr()
df_corr
df_corr = df_corr[:-1]
df_corr

max = abs(df_corr).max()
min = abs(df_corr).min()
a = max + min
round(a, 2)

#당뇨병 예측 모델 실습
import pandas as pd

train = pd.read_csv('C:/Users/user/codingstudy/data/diabetes_train.csv')
test = pd.read_csv('C:/Users/user/codingstudy/data/diabetes_test.csv')

train.head()
test.head()

train.info()

train = train.drop('id', axis=1)
train = train.drop('Outcome', axis=1)
test = test.drop('id', axis=1)

y_train = train['Outcome']
y_train

train.shape, y_train.shape

from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier(n_estimators = 300, random_state = 2021, max_depth=7)

model.fit(train, y_train)

#model.predict() -> 최종 예측값 반환
#model.predict_proba() -> 각 클래스에 대한 확률값 반환
#model.predict_log_proba() -> 각 클래스 확률의 로그값 반환
prediction = model.predict_proba(test)
model.score(train, y_train)
prediction

#특정 클래스 확률값 출력
print(model.classes_)

output = pd.DataFrame({'결과' : prediction[:,1]})
output


#조건에 따른 상위 값 전처리
#city와 f4를 기준으로 f5의 평균값을 구한 다음,
#f5를 기준으로 상위 7개 값을 모두 더해 출력하시오 (소수점 둘째자리까지 출력)
df = pd.read_csv('C:/Users/user/codingstudy/data/basic1.csv')
df
df1 = df.groupby(['city', 'f4'])['f5'].mean()
#reset_index()를 통해 dataframe화
df1 = df1.reset_index().sort_values('f5', ascending=False).head(7)
round(sum(df1['f5']), 2)


#슬라이싱, 조건 실기
# 주어진 데이터 셋에서 age컬럼 상위 20개의 데이터를 구한 다음 
# f1의 결측치를 중앙값으로 채운다.
# 그리고 f4가 ISFJ와 f5가 20 이상인 
# f1의 평균값을 출력하시오!

df = pd.read_csv('C:/Users/user/codingstudy/data/basic1.csv')
df

#상위 20개 데이터 추출
df = df.sort_values(['age'], ascending=False).head(20)
df

#결측치 중앙값으로 채우기
df.isnull().sum()
mid = df['f1'].median()
mid
df['f1'] = df['f1'].fillna(mid)
df.isnull().sum()

#f4가 ISFJ, f5가 20이상인 행 조회
df[(df['f4'] == 'ISFJ') & (df['f5'] >= 20)]['f1'].mean()

#분산
# 주어진 데이터 셋에서 f2가 0값인 데이터를 age를 기준으로 오름차순 정렬하고
# 앞에서 부터 20개의 데이터를 추출한 후 
# f1 결측치(최소값)를 채우기 전과 후의 분산 차이를 계산하시오 (소수점 둘째 자리까지)
df = pd.read_csv('C:/Users/user/codingstudy/data/basic1.csv')
df

df1 = df[df['f2'] == 0]
df1.sort_values('age', ascending=True)
df1 = df1.head(20)
df1.isnull().sum()
bf = round(df1['f1'].var(),2)
bf

df['f1'] = df['f1'].fillna(df['f1'].min())
df.isnull().sum()

bf2 = round(df['f1'].var(),2)
bf2

#시계열 데이터 실습
import datetime
import pandas as pd
df = pd.read_csv('C:/Users/user/codingstudy/data/basic2.csv')
df.info()
# datetime으로 type변경
df['Date'] = pd.to_datetime(df['Date'])
df.info()
#연, 월, 일 추가
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df
#2022년 5월 sales의 중앙값을 구하시오
df[(df['year'] == 2022) & (df['month'] == 5)]['Sales'].median()

