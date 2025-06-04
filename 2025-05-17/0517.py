# 데이터프레임 확인
import pandas as pd

df = pd.read_csv('C:/Users/user/codingstudy/data/Customer_Data.csv')
print(df.head())

print(df.columns)
print(df.describe())
print(df.shape)

# 데이터프레임 인덱싱 및 슬라이싱

df[['age', 'income']]
df.loc[10:15, ['age', 'income']]
df.loc[:, ['age']]
print(df.loc[25, ['age']])

#조건으로 행 선택
print(df.loc[df['income'] >= 50000])
print(df.groupby(df['income'] >= 50000))

print(df.loc[df['income'] >= 50000], df[['age', 'income']])
print(df.loc[df['age'] < 30, ['age']])

#age가 30 이상인 모든 열
df.loc[df['age'] >= 30]
#age가 30이상인 모든 age, income 데이터
print(df.loc[df['age'] >= 30][['age', 'income']])
#age가 최댓값을 가지는 행
cond3 = df['age'] == df['age'].max() # 1차원 series형식으로 True, False 출력
print(df[cond3])
print(df[df['age'] == df['age'].max()]) # 조건에 따라 True인 행 출력

#age 가 20대 이상, 50대 미만 행 출력
cond1 = df['age'] >= 20
cond2 = df['age'] < 50
print(df[(cond1) & (cond2)])

#해당 조건의 특정 열 출력
df[(df['age'] >= 20) & (df['age'] < 50)][['age', 'income']]
df.loc[(df['age'] >= 20) & (df['age'] < 50), ['age', 'income']]

#iloc(데이터가 있는 숫자로 위치 접근)
#조건식 불가
print(df.iloc[0])
print(df.iloc[-1])

print(df.iloc[:,0])
print(df.iloc[:,-1])

#첫 4개 행, 열
print(df.iloc[0:4])
print(df.iloc[:,0:4])

#1~2번째 까지 열의 1, 3행 출력
print(df.iloc[[1 ,3], 0:2])
