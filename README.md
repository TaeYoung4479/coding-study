# Coding-study
빅분기 실기 공부방

2025/05/15 시작

## 파이썬 기초

데이터 프레임에서 개수를 확인할 때는 

### value_counts()
```python
menu = df['menu'].value_counts()
menu
```
```python
아메리카노     3
딸기라떼      1
카페라떼      1
바닐라라떼     1
초코라떼      1
모카라떼      1
카라멜라떼     1
챠이 라떼     1
오곡 라떼     1
토피넛 라떼    1
카페 라떼     1
Name: count, dtype: int64
```

-> Series는 1차원 데이터, DataFrame은 2차원 데이터

-> 데이터프레임에 대한 크기 확인 -> df.shape()

-> 데이터프레임 기술통계량 확인 -> df.describe()


2025/05/17

### 데이터프레임 인덱싱 및 슬라이싱

특정 열선택 -> df['col_name'] / df.col_name
여러 열 선택 가능 -> df[['col_name_1','col_name_2']]

특정 행 선택 -> df.loc[행시작:행 끝, ['열이름1', '열이름2']]
'''python

'''
