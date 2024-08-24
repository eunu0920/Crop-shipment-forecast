from neuralprophet import NeuralProphet, set_log_level
set_log_level("ERROR")
import pandas as pd
from sklearn import preprocessing
import warnings 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib import pyplot as plt
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from neuralprophet import load, save
# 데이터 로드
file_path = r'C:\Users\kimeu\OneDrive\사진\바탕 화면\농작물_웹페이지\pages2\cucumber_hongcheon.xlsx'
data = pd.read_excel(file_path, sheet_name=0)

# 결측값 0으로 채우기
data.fillna(0, inplace=True)

# 데이터 전처리
data['y'] = data['y'].astype(int)
data['ds'] = pd.to_datetime(data['ds'])
data=data.drop(['sn3','ya3','wo3'], axis=1)

# 외부 변수 칼럼 리스트 설정
col_lst = ['temp', 'max_Temp',
       'min_Temp', 'hum', 'widdir', 'wind', 'rain',
       'sun_Time', 'sun_Qy', 'condens_Time', 'gr_Temp']

# train test split
cutoff = "2023-08-01" #데이터 분할 기준
train = data[data['ds']<cutoff]
validate = data[data['ds']>=cutoff]

le = preprocessing.LabelEncoder()

################# Create Model ####################
m = NeuralProphet(
    n_lags = 1*31, 
    n_forecasts = 1*31,
    weekly_seasonality='auto', 
    yearly_seasonality='auto', 
    daily_seasonality=False, # 이 위에는 건들면 안됨!!!!
    learning_rate=0.02, #학습률 설정
    batch_size=30, #배치 사이즈 설정
    epochs=1 #학습횟수
)
# 학습률, 배치사이즈, 학습횟수 바꿔가며 가장 성능이 좋은 값 찾기

# 모델 학습
m = m.add_lagged_regressor(names=col_lst, normalize="soft")
warnings.filterwarnings("ignore")
metrics = m.fit(data ,freq='D', validation_df=validate, progress='plot')

future_df = m.make_future_dataframe(data, periods=31, n_historic_predictions=True)
forecast_future = m.predict(future_df)

#yhat1과 실제값 시각화
forecast = m.predict(validate)
fig = m.plot(forecast, [['ds', 'y', 'yhat1']])

# Select only columns starting with "yhat"
yhats = forecast.filter(regex='^yhat')

# Reshape data into long format
melted_yhats = yhats.melt(var_name='yhat', value_name='value', ignore_index=False)

forecast.fillna(0,inplace=True)
forecast['yhat1'] = forecast['yhat1'].apply(np.int64)

forecast['yhat1'] [forecast['yhat1'] <0]=0

mae = mean_absolute_error(forecast['y'], forecast['yhat1'])
rmse = np.sqrt(mean_squared_error(forecast['y'], forecast['yhat1']))
print(' mae=', mae, '\n', 'RMSE=', rmse)

future = m.make_future_dataframe(data, periods=1*31, n_historic_predictions=True)
forecast_future = m.predict(future)

def exctract_yhat(forecast_future, size= 1*31):
    columns = forecast_future.columns[3:]
    newframe = forecast_future[['ds', 'yhat1']].iloc[-size:].copy()
    for col in columns:
        if 'yhat' in col:
            newframe['yhat1'] = newframe['yhat1'].fillna(forecast_future[col])
    return newframe

file_path="C:/Users/kimeu/OneDrive/사진/바탕 화면/농작물_웹페이지/pages2"
save(m, 'cucumber_hongcheon.np')
