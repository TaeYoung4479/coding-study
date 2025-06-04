# 데이터프레임 확인
import pandas as pd

df = pd.read_csv('C:/Users/user/codingstudy/data/Customer_Data.csv')
print(df.head())

#열 삭제
#axis = 1 -> 열 기준 삭제, inplace=True 옵션 가능\
#axis =0 -> 행 기준 삭제
df_1 = df.drop(['income'], axis=1)
df_1
#행 인덱스 번호로 삭제
df_2 = df.drop(499, axis=0)
df_2

#컬럼 이름 변경
df_1 = df.rename(columns={'income' : '소득'})
df_1

#날짜 형식 데이터 처리
df = pd.read_csv('C:/Users/user/codingstudy/data/website.csv')
print(df.head())

df.info()

#object형식의 데이터 날자 형식으로 변환
import datetime
#람다 함수를 통해 전체 x를 다음과 같이 변환
df['StartTime'] = df['StartTime'].apply(lambda x: pd.to_datetime(str(x), format='%Y-%m-%d %H:%M:%S'))
df.info()

#datetime함수를 통해 바로 변환 가능
df['EndTime'] = pd.to_datetime(df['EndTime'])
df.info()

#datetime 라이브러리를 통해 추출 가능
df.iloc[10, df['EndTime']].year

#데이터 가공 및 그룹핑