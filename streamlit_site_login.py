import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from streamlit_extras.app_logo import add_logo
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from neuralprophet import NeuralProphet, set_log_level
from neuralprophet import save, load, set_random_seed
from dateutil import parser
from io import StringIO
from pytorch_lightning.loggers import TensorBoardLogger
from tkinter.tix import COLUMN
from pyparsing import empty
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns
import xlsxwriter
import datetime
import warnings
import pickle
import time
import io
import os
import folium
from streamlit_folium import folium_static
from PIL import Image
import base64
from io import BytesIO
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# 페이지 설정 - 가장 먼저 호출
st.set_page_config(
    page_title='농작물 출하량 예측',
    page_icon="👋",
    layout='wide'
)

# 로컬 이미지를 Base64로 인코딩하여 반환하는 함수
def get_base64_of_image(image_file):
    if not os.path.isfile(image_file):
        raise FileNotFoundError(f"{image_file} 파일을 찾을 수 없습니다.")
    img = Image.open(image_file)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# 농작물별 재배지와 예측 출하량 데이터
def get_crop_data(crop):
    data = {
        '오이': {
            '천안': [36.8258, 127.1131], 
            '공주': [36.4588, 127.2445], 
            '진천': [36.8969, 127.4233], 
            '춘천': [37.8804, 127.7292],
            '홍천': [37.7584, 127.8014]
        },
        '토마토': {
            '철원': [38.2602, 127.2763], 
            '보성': [34.7683, 127.0821], 
            '장수': [35.6443, 127.5233]
        },
        '파프리카': {
            '철원': [38.2602, 127.2763], 
            '진주': [35.2064, 128.0971], 
            '장흥': [34.6511, 126.9569], 
            '창녕': [35.5682, 128.2772]
        }
    }
    return data.get(crop)

# 지도 생성 함수
def create_map(crop_locations, selected_region):
    # 대한민국 지도 기본 설정
    korea_map = folium.Map(location=[36.5, 127.5], zoom_start=7)

    # 농작물 재배지 마커 추가
    for region, (lat, lon) in crop_locations.items():
        # 지역별 원의 크기 설정
        if region == "천안":
            radius = 23.5
        elif region == "공주":
            radius = 1.6
        elif region == "진천":
            radius = 5.9
        elif region == "춘천":
            radius = 16.7
        elif region == "홍천":
            radius = 17
        else:
            radius = 10  # 기본값

        color = 'red' if region == selected_region else 'blue'  # 선택한 지역은 빨간색, 나머지는 파란색
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,  # 지역별로 설정된 원의 크기
            color=color,  # 외곽선 색상
            fill=True,  # 원을 채울지 여부
            fill_color=color,  # 채우기 색상
            fill_opacity=0.7,  # 채우기 투명도
            popup=f"{region} (선택된 지역)" if region == selected_region else f"{region}"
        ).add_to(korea_map)

    return korea_map

kgdata = pd.read_excel("C:/Users/kimeu/OneDrive/사진/바탕 화면/kg.xlsx", sheet_name=0)
forecast = pd.read_excel("C:/Users/kimeu/OneDrive/사진/바탕 화면/forecast.xlsx", sheet_name=0)

image = Image.open("C:/Users/kimeu/OneDrive/사진/바탕 화면/site/titlelogo.png")

# Streamlit 앱 설정
def main():
  # 로고 이미지 파일 경로
    logo_path = "C:/Users/kimeu/OneDrive/사진/바탕 화면/site/logo.png"

    # 상단에 로고 추가
    try:
        base64_image = get_base64_of_image(logo_path)
        st.markdown(
            """
            <style>
            .logo {
                display: flex;
                justify-content: center;
                align-items: center;
                margin-bottom: 20px;
            }
            .logo img {
                width: 100px; /* 원하는 로고 크기로 조절 */
                height: auto;
            }
            </style>
            <div class="logo">
                <img src="data:image/png;base64,%s" alt="로고">
            </div>
            """ % base64_image,
            unsafe_allow_html=True
        )
    except FileNotFoundError as e:
        st.error(f"로고 파일을 찾을 수 없습니다: {e}")

  

# YAML 파일 읽기
with open('config.yaml') as file:
    config = yaml.load(file, Loader=stauth.SafeLoader)

# yaml 파일 데이터로 객체 생성
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    int(config['cookie']['expiry_days']),  # 이 값을 정수로 변환
    config['preauthorized']
)

# 로그인 위젯 렌더링
name, authentication_status, username = authenticator.login('main')

# 로그인 상태 확인
if authentication_status:
    # 로그인 성공 후 로그아웃 버튼 표시
    authenticator.logout("Logout", "sidebar")
    
    # 사용자의 역할에 따라 페이지 표시
    if username == 'farmer':
        st.sidebar.title(f"환영합니다, {name}님!")
        st.sidebar.title('메뉴')
        selected_option = st.sidebar.radio("메뉴 선택", ["홈", "농작물 출하량 예측", "예측 출하량 지도", "수급 현황 분석"])
    

        if selected_option == "홈":
            st.title('🌿GREENINSIGHT')
            st.write("")
            st.write('GREENINSIGHT는 최적의 Neural Prophet 모델을 사용하여 농작물 출하량 예측 결과를 제공합니다.')
            st.write('농작물 출하량 예측 페이지로 이동하려면 사이드바에서 "농작물 출하량 예측"을 선택하세요.')
            st.write("")
            st.write("")
            st.write("")
            st.image("C:/Users/kimeu/OneDrive/사진/바탕 화면/organic-farming-concept.png", use_column_width=True)  
            st.write("")  
            st.write("")                 
            st.write("")            
            st.write("")
            st.write("")   
            st.write("")  
            st.write("")       
            st.write("")          
            st.write("")            
            st.write("")            
                       
            
############################################################# 1. 농작물 출하량 예측 ###########################################################

        elif selected_option == "농작물 출하량 예측":
            # 농작물 선택
            selected_crop = st.sidebar.selectbox('작물을 선택하세요: ', ['오이', '토마토', '파프리카', '양파', '배추', 
                                                                 '당근', '고추', '옥수수', '무', '대파', '사과', '배', '호박', '가지', 
                                                                 '마늘', '포도', '복숭아', '자두', '쌀', '감자'])

            if selected_crop == '오이':
                st.subheader('오이 출하량 예측')
                st.write('오이의 출하량 예측과 주요 재배지 정보를 제공합니다.')

            elif selected_crop == '토마토':
                st.subheader('토마토 출하량 예측')
                st.write('토마토의 출하량 예측과 주요 재배지 정보를 제공합니다.')

            elif selected_crop == '파프리카':
                st.subheader('파프리카 출하량 예측')
                st.write('파프리카의 출하량 예측과 주요 재배지 정보를 제공합니다.')

            # 선택한 농작물에 대한 데이터 가져오기
            crop_locations = get_crop_data(selected_crop)

            # 재배지 선택
            selected_region = st.selectbox('출하량 예측을 진행할 재배지를 선택하세요:', list(crop_locations.keys()))

            # 출하지, 작물 영어이름 설정
            def translate_crop_name(selected_crop):
                translation_dict = {
                '오이': 'cucumber',
                '토마토': 'tomato',
                '파프리카': 'paprika'
                }
                return translation_dict.get(selected_crop, selected_crop)  # 기본적으로 원래 값 반환

            def translate_region_name(selected_region):
                translation_dict = {
                '천안': 'cheonan',
                '춘천': 'chuncheon',
                '공주': 'gongju',
                '홍천': 'hongcheon',
                '진천': 'jincheon',
                '보성': 'bosang',
                '철원': 'charan',
                '장수': 'jangsu',
                '진주': 'jinju',
                '장흥': 'jangheung',
                '창녕': 'cangnyeong'
                }
                return translation_dict.get(selected_region, selected_region)  # 기본적으로 원래 값 반환

            translated_crop = translate_crop_name(selected_crop)
            translated_region = translate_region_name(selected_region)

            # 파일 경로에 f-string 사용
            data = pd.read_excel(f'C:/Users/kimeu/OneDrive/사진/바탕 화면/농작물_웹페이지/pages2/{translated_crop}_{translated_region}.xlsx', sheet_name=0)

            # 데이터 확인
            data['y'] = data['y'].astype(int)
            data['ds'] = pd.to_datetime(data['ds'])
            data=data.drop(['sn3','ya3','wo3'], axis=1)

            
            st.write(f'- {selected_region}에서 재배되는 {selected_crop}의 미래 출하량을 예측합니다.')
            st.markdown('- 날짜를 입력하고 **:blue[예측버튼]**을 클릭하면 이후 경매물량이 예측됩니다.')
            st.write("")
            st.write("")

            with st.form('Area-form'):
                d = st.date_input("언제부터 경매물량을 예측하시겠습니까?", datetime.date.today())
                st.write('선택한 날짜:', d)
                submitted = st.form_submit_button("예측하기")

                if submitted:
                    st.write("")
                    st.write("")
                    st.write('#### ■ 예측결과:')
                    warnings.filterwarnings("ignore")
                    with st.spinner('모델 로딩 중입니다.'):
                        m_neural_prophet = load(f"C:/Users/kimeu/OneDrive/사진/바탕 화면/농작물_웹페이지/pages2/{translated_crop}_{translated_region}.np")
                        time.sleep(3)
                    
                    st.success('예측 완료!')

                    # 선택한 날짜(d)로부터 31일 전의 과거 출하량 합계를 계산
                    start_date_past = pd.Timestamp(d) - pd.DateOffset(days=31)
                    end_date_past = pd.Timestamp(d) - pd.DateOffset(days=1)

                    # 날짜 범위에 해당하는 데이터 필터링
                    filtered_past_data = data[(data['ds'] >= start_date_past) & (data['ds'] <= end_date_past)]

                    # 과거 출하량 합계 계산
                    past_sum = filtered_past_data['y'].sum()
                    days = 31

                    # 선택한 날짜(d)로부터 31일 동안의 future_df 생성
                    future_df = m_neural_prophet.make_future_dataframe(
                        df=data, 
                        periods=days,  # 예측할 기간 (31일)
                        n_historic_predictions=True
                    )

                    forecast_future = m_neural_prophet.predict(future_df)

                    def exctract_yhat(forecast_future, size=31):
                        future_predictions = forecast_future[['ds', 'yhat1']]
                        future_predictions = future_predictions[future_predictions['ds'] > d]
                        future_predictions = future_predictions.head(size)  # 특정 일수만큼 가져옴
                        return future_predictions

                    def exctract_yhat(forecast_future, size=31):
                        columns = forecast_future.columns[3:]
                        newframe = forecast_future[['ds', 'yhat1']].iloc[-size:].copy()
                        for col in columns:
                            if 'yhat' in col:
                                newframe['yhat1'].update(forecast_future[col])
                        return newframe

                    result = exctract_yhat(forecast_future)
                    result.fillna(0, inplace=True)
                    result['yhat1'] = result['yhat1'].apply(np.int64)
                    result['yhat1'][result['yhat1'] < 0] = 0
                    result = result[['ds', 'yhat1']]

                    # 과거 출하량 합계 계산 (이미 숫자 형식임)
                    past_sum = filtered_past_data['y'].sum()

                    # 예측 출하량 합계 계산 (포맷팅 전 숫자 상태로 유지)
                    future_sum = result['yhat1'].sum()
                    future_sum1 = result['yhat1'].iloc[:7].sum()
                    future_sum2 = result['yhat1'].iloc[:14].sum()

                    # 차이 계산 (숫자끼리의 뺄셈)
                    difference = future_sum - past_sum

                    # 추세 메시지 설정
                    if future_sum < past_sum:
                        trend_message = f"'{selected_region}'지역 출하량이 '감소'할 전망입니다."
                    else:
                        trend_message = f"'{selected_region}'지역 출하량이 '증가'할 전망입니다."

                    # 각 값을 천 단위 콤마 추가하여 포맷팅
                    future_sum = "{:,}".format(future_sum)
                    future_sum1 = "{:,}".format(future_sum1)
                    future_sum2 = "{:,}".format(future_sum2)
                    difference = "{:,}".format(difference)

                    st.write("")
                    st.write("")     
                    # Streamlit에서 출력
                    col1, col2, col3 = st.columns(3)
                    col1.metric("<미래 '7️⃣일' 예측 출하량 합계>",
                                f"{future_sum1}kg")
                    col2.metric("<미래 '1️⃣4️⃣일' 예측 출하량 합계>",
                                f"{future_sum2}kg")
                    col3.metric("<미래 '3️⃣1️⃣일' 예측 출하량 합계>",
                                f"{future_sum}kg")
                    st.write("") 
                    st.write("") 
                    st.metric("#### ● 이후 한 달 동안의 예측 추세", trend_message, f"{difference}kg")
                    st.write("")
                    st.write("") 
                    st.write("")
                    
                    # 예시이미지 추가
                    st.image("C:/Users/kimeu/OneDrive/사진/바탕 화면/농작물_웹페이지/pages2/image.png", width=300)
                    st.write("")
                    st.write("")
                    st.write("")  
                    st.write("#### ● 예측된 데이터의 추세 그래프")
                    fig_forecast = px.line(result, x='ds', y='yhat1')
                    fig_forecast.update_xaxes(range=[d, d + datetime.timedelta(days=31)])

                    st.plotly_chart(fig_forecast, use_container_width=True)
                    st.write("#### ● 실제 출하량의 추세 그래프")
                    data_real_value = data[['ds', 'y']].rename(columns={'ds': '날짜', 'y': '실제 출하량'})
                    fig_real = px.line(data_real_value, x='날짜', y='실제 출하량')
                    st.plotly_chart(fig_real, use_container_width=True)

################################################ 2. 예측 출하량 지도 #####################################################

        elif selected_option == "예측 출하량 지도":
            # 농작물 선택
            selected_crop = st.sidebar.selectbox('작물을 선택하세요: ', ['오이', '토마토', '파프리카', '양파', '배추', 
                                                                 '당근', '고추', '옥수수', '무', '대파', '사과', '배', '호박', '가지', 
                                                                 '마늘', '포도', '복숭아', '자두', '쌀', '감자'])


            # 선택한 농작물에 대한 데이터 가져오기
            crop_locations = get_crop_data(selected_crop)
            st.subheader(f'{selected_crop}의 예측 출하량 지도')
            st.write(f'{selected_crop}의 지역별 출하량 예측 결과를 지도에 표시합니다.')

            # 재배지 선택
            selected_region = st.selectbox('표시할 재배지를 선택하세요:', list(crop_locations.keys()))

            # 지도 생성 및 표시
            st.write("")
            korea_map = create_map(crop_locations, selected_region)
            folium_static(korea_map, width=600, height=400)           

################################################## 3. 수급 현황 분석 ########################################################

        elif selected_option == "수급 현황 분석":
            selected_crop = st.sidebar.selectbox('작물을 선택하세요: ', ['오이', '토마토', '파프리카', '양파', '배추', '당근', '고추', '옥수수', 
                                                                 '무', '대파', '사과', '배', '호박', '가지', '마늘', '포도', '복숭아', '자두', '쌀', '감자'])

            # 선택한 농작물에 대한 데이터 가져오기
            crop_locations = get_crop_data(selected_crop)

            st.subheader(f'{selected_crop}의 수급 현황 분석') # ex 오이의 출하량은 이후 한달 기준 물량에 50kg 초과될 전망입니다. 
            st.write(f'{selected_crop}의 기준 물량 대비 예측 출하량 정보를 제공합니다.')

            # 날짜 선택 위젯
            selected_date = st.date_input("날짜를 선택하세요:", value=forecast['ds'].min())

            # 선택한 날짜를 datetime 형식으로 변환하여 시, 분, 초를 제거
            selected_date = pd.to_datetime(selected_date).normalize()

            # 선택한 날짜에 해당하는 데이터 가져오기
            selected_forecast = forecast[forecast['ds'].dt.normalize() == selected_date]
            selected_kgdata = kgdata[kgdata['ds'].dt.normalize() == selected_date]

            # 예측량과 기준 수급물량 비교
            if not selected_forecast.empty and not selected_kgdata.empty:
                yhat1_value = selected_forecast['yhat1'].values[0]
                kg_value = selected_kgdata['kg'].values[0]
                difference = abs(yhat1_value - kg_value)
                formatted_difference = f"{difference:,.0f}"  # 1000단위 콤마 추가 및 소수점 제거

                if yhat1_value > kg_value:
                    trend_message = f"<span style='font-size:24px;'> '{selected_date.date()}'에 예측 출하량이 기준 수급물량보다 <span style='color:blue;'>{formatted_difference}kg</span> 초과될 것으로 전망됩니다.</span>"
                elif yhat1_value < kg_value:
                    trend_message = f"<span style='font-size:24px;'> '{selected_date.date()}'에 예측 출하량이 기준 수급물량보다 <span style='color:red;'>{formatted_difference}kg</span> 미달될 것으로 전망됩니다.</span>"
                else:
                    trend_message = f"<span style='font-size:24px;'> '{selected_date.date()}'에 예측 출하량과 기준 수급물량이 동일할 것으로 전망됩니다.</span>"

                # Streamlit에서 HTML을 활용하여 색상 및 포맷 적용
                st.markdown(trend_message, unsafe_allow_html=True)
            else:
                st.write(f"선택한 날짜 {selected_date.date()}에 대한 데이터가 없습니다.")

            # 그래프 그리기
            fig = px.line(forecast, x="ds", y="yhat1", color_discrete_sequence=["#0514C0"], labels={'y': 'forecast'})
            fig.add_scatter(x=kgdata['ds'], y=kgdata['kg'], mode='lines', name='기준 수급물량', line=dict(color='#4CC005'))
            st.plotly_chart(fig, use_container_width=True)

        st.sidebar.write("")
        st.sidebar.write("")
        st.sidebar.write("")
        st.sidebar.write("")
        st.sidebar.write("")
        st.sidebar.write("")
        st.sidebar.write("") 
        st.sidebar.write("")  
        st.sidebar.write("")
        st.sidebar.write("") 
        st.sidebar.write("")                             
        st.sidebar.image(image)            
 
elif authentication_status == False:
    # 로그인 실패 시 메시지 표시
    st.error("ID나 비밀번호가 다릅니다. 다시 확인해주세요.")

elif authentication_status == None:
    # 로그인 필요 메시지 표시
    st.warning("ID와 비밀번호를 입력하세요.")

if __name__ == "__main__":
    if authentication_status:  # 로그인 성공 시만 main 함수 실행
        main()
