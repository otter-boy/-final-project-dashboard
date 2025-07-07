import streamlit as st

import gdown
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

import openai
from openai import OpenAI

st.set_page_config(layout='wide')


# ======= basic df ===========
@st.cache_data
def load_data():
    df_basic = pd.read_csv("./data/df_basic.csv")
    return df_basic

df_basic = load_data()

# ====== plus df =========
# 명서님 - 시간대별 분석용

@st.cache_data
def load_plus_t():
    return pd.read_csv("./data/plus_t.csv")

plus_t = load_plus_t()

# 민주님 - 광고분석용
df_ad = pd.read_csv('./data/광고상품ROI.csv')

# 경민님 - 버블차트용
open_df = pd.read_csv('./data/open_df.csv', encoding='utf-8-sig')

# 경민님 - 상품 클러스터링용
cluster_df = pd.read_csv('./data/cluster_df.csv')

# ===================
# pro 대시보드에 들어가는 데이터프레임

@st.cache_data
def load_dashboard_pro():
    df_pro = pd.read_csv("./data/df_dashboard_pro.csv")
    # 'Untitled' 제외한 스토어명 익명화
    unique_sellers = [name for name in df_pro['스토어명'].unique() if name != 'untitled']
    seller_map = {name: f'셀러{str(i+1).zfill(4)}' for i, name in enumerate(unique_sellers)}

    # 이름 매핑하기
    df_pro['스토어명'] = df_pro['스토어명'].apply(lambda x: seller_map.get(x, x))
    return df_pro

df_pro = load_dashboard_pro()

word = pd.read_csv('./data/word.csv')

# ===================
# 로그인 여부
if 'login' not in st.session_state:
    st.session_state.login = False

# 셀러명 입력
if 'seller_name' not in st.session_state:
    st.session_state.seller_name = ''

if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# 사이드바 관리
with st.sidebar:
    
    if st.session_state.page == 'Home': st.markdown('# 대시보드 홈')
    elif st.session_state.page == 'Basic': st.markdown('# Basic 대시보드')
    elif st.session_state.page == 'Plus': st.markdown('# Plus 대시보드')
    elif st.session_state.page == 'Pro': st.markdown('# Pro 대시보드')
    
    # 로그인 전
    if not st.session_state.login:
    # 셀러명 입력
        seller_input = st.text_input('셀러명을 입력하세요', placeholder='셀러0001')
        if st.button('로그인'):
            if seller_input in df_pro['스토어명'].to_list():
                st.session_state.login = True
                st.session_state.seller_name = seller_input
                st.session_state.page = 'Home'
                st.rerun()
            else:
                st.warning('❌셀러명을 확인해주세요❗️ 등록되지 않은 셀러입니다.')
    # 로그인 후
    if st.session_state.login:
    
        st.success(f'👋 **{st.session_state.seller_name}**님 환영합니다!')
        
        if st.button('로그아웃'):
            st.session_state.login = False
            st.session_state.seller_name = ''
            st.session_state.page = 'Home'
            st.rerun()
        
        st.divider()
        
        st.header('대시보드 선택')
        # 대시보드 선택 메뉴 버튼
        if st.button('Home', use_container_width=True): 
            st.session_state.page = 'Home'
            st.rerun()
        if st.button('Basic', use_container_width=True): 
            st.session_state.page = 'Basic'
            st.rerun()
        if st.button('Plus', use_container_width=True): 
            st.session_state.page = 'Plus'
            st.rerun()
        if st.button('Pro', use_container_width=True): 
            st.session_state.page = 'Pro'
            st.rerun()
    else:
        st.markdown('🔐 로그인 후 메뉴가 표시됩니다.')
        
        
# 대시보드 홈 page
if st.session_state.page == 'Home':

    st.title('대시보드 홈')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div style='background-color:#E8F5E9; padding:20px; border-radius:20px;
                        box-shadow:4px 4px 5px rgba(0,0,0,0.05); text-align:center; font-weight:bold; min-height:300px;'>
                <div style='font-size:45px;'>
                    Basic
                </div>
                <div style="font-size:18px; fong-weight:bold; margin-bottom:20px; line-height: 3; text-align:left;">
                    ✔️ 월별 방송 실적 요약  
                    <br>✔️ 주요 트렌드 제공 
                    <br>✔️ 방송 동향 시각화
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.session_state.login:
            st.write("")
            st.button("**Basic 대시보드로 이동**", key="basic", use_container_width=True, on_click=lambda: st.session_state.update({"page": "Basic"}))

    with col2:
        
        st.markdown(
            """
            <div style='background-color:#C8E6C9; padding:20px; border-radius:20px;
                        box-shadow:4px 4px 5px rgba(0,0,0,0.05); text-align:center; font-weight:bold; min-height:300px;'>
                <div style='font-weight:bold; font-size:45px;'>
                    Plus
                </div>
                <div style="font-size:18px; fong-weight:bold; margin-bottom:20px; line-height: 3; text-align:left;">
                    ✔️ 카테고리별 인사이트 기반
                    <br>✔️ 실전형 분석 도구 제공 
                    <br>✔️ 성과 향상 방향 제시 
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.session_state.login:
            st.write("")
            st.button("**Plus 대시보드로 이동**", key="plus", use_container_width=True, on_click=lambda: st.session_state.update({"page": "Plus"}))

    with col3:
        st.markdown(
            """
            <div style='background-color:#A5D6A7; padding:20px; border-radius:20px;
                        box-shadow:4px 4px 5px rgba(0,0,0,0.05); text-align:center; font-weight:bold; min-height:300px;'>
                <div style='font-weight:bold; font-size:45px;'>
                    Pro
                </div>
                <div style="font-size:18px; fong-weight:bold; margin-bottom:20px; line-height: 3; text-align:left;">
                    ✔️ 셀러 개인별 맞춤 피드백 제공  
                    <br>✔️ 방송 전략 제안  
                    <br>✔️ 효율적인 운영 인사이트 제공
                </div>
            """,
            unsafe_allow_html=True
        )
        if st.session_state.login:
            st.write("")
            st.button("**Pro 대시보드로 이동**", key="pro", use_container_width=True, on_click=lambda: st.session_state.update({"page": "Pro"}))
# Basic 대시보드 page
elif st.session_state.page == 'Basic':

    st.title('Basic 대시보드')
    st.caption('월별 방송 실적 요약과 주요 트렌드를 간단하게 제공합니다.')
    st.markdown("""
    <div style='background-color:#DFF5E1; padding:10px; border-radius:8px;'>
    📅 <b>분석 기간 : 2024년 6월 ~ 2025년 5월</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    years = [2024, 2025]

    # 초기값 설정
    default_year = 2024
    default_month = 6

    col1, col2 = st.columns(2)
    with col1:
        selected_year = st.selectbox("연도 선택", years, index=0)
    
    if selected_year == 2024:
        months = [6, 7, 8, 9, 10, 11, 12]

    if selected_year == 2025:
        months = [1, 2, 3, 4, 5]
        
    with col2:
        selected_month = st.selectbox("월 선택", months, index=0)
            

    df_b = df_basic[
        (df_basic['시작시간'].dt.year == selected_year) &
        (df_basic['시작시간'].dt.month == selected_month)
    ]

    # ===== 핵심 성과 지표 =====
    매출 = int(df_b['총 매출액(원)'].sum())
    전환율 = df_b['구매 전환율'].mean()
    조회수 = int(df_b['방송조회수'].sum())
    판매자수 = df_b['스토어명'].nunique()
    방송수 = df_b.shape[0]
    
    # 전월 데이터 추출
    curr_year = selected_year
    curr_month = selected_month

    if curr_month == 1:
        prev_year = curr_year - 1
        prev_month = 12
    else:
        prev_year = curr_year
        prev_month = curr_month - 1

    df_prev = df_basic[
        (df_basic['시작시간'].dt.year == prev_year) &
        (df_basic['시작시간'].dt.month == prev_month)
    ]

    if not df_prev.empty:
        prev_매출 = int(df_prev['총 매출액(원)'].sum())
        prev_전환율 = df_prev['구매 전환율'].mean()
        prev_조회수 = int(df_prev['방송조회수'].sum())
        prev_판매자수 = df_prev['스토어명'].nunique()
        prev_방송수 = df_prev.shape[0]
    else:
        prev_매출 = prev_전환율 = prev_조회수 = prev_판매자수 = prev_방송수 = None

    def get_delta_text(curr, prev, unit="%"):
        if prev is None or prev == 0:
            return ""
        diff = curr - prev
        rate = (diff / prev) * 100
        
        if rate > 0:
            arrow = "▲"
            color = "#d32f2f"
        elif rate < 0:
            arrow = "▼"
            color = "#007acc"
        else:
            arrow = "⏺️"
            color = "#888888"
            
        color = "#d32f2f" if rate > 0 else "#1565c0" if rate < 0 else "#888"
        return f"<div style='font-size:16px; color:{color}; margin-top:5px;'>{arrow} {abs(rate):.1f}{unit}</div>"
  
    
    st.markdown("""
    <div style='background-color:#DFF5E1; padding:10px 15px; border-radius:8px; margin-bottom:10px;'>
    <span style='font-weight:bold; font-size:22px;'>📊 핵심 성과 지표</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div style="background-color:#F3FBF5; padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
            <div style="font-size:16px;">매출액</div>
            <div style="font-size:28px; font-weight:bold;">{매출 / 1e8:.1f}억</div>
            {get_delta_text(매출, prev_매출)}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background-color:#F3FBF5; padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
            <div style="font-size:16px;">구매 전환율</div>
            <div style="font-size:28px; font-weight:bold;">{전환율:.3f}%</div>
            {get_delta_text(전환율, prev_전환율)}
        </div>
        """, unsafe_allow_html=True)

    # 조회수 표현 방식 결정
    조회수_단위 = 0
    
    if 조회수 >= 1e8:
        조회수_단위 = f'{조회수 / 1e8:.2f}억'
    else:
        조회수_단위 = f'{조회수 / 1e4:.0f}만'
    with col3:
    
        st.markdown(f"""
        <div style="background-color:#F3FBF5; padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
            <div style="font-size:16px;">방송 조회수</div>
            <div style="font-size:28px; font-weight:bold;">{조회수_단위}</div>
            {get_delta_text(조회수, prev_조회수)}
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style="background-color:#F3FBF5; padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
            <div style="font-size:16px;">판매자 수</div>
            <div style="font-size:28px; font-weight:bold;">{판매자수:,}</div>
            {get_delta_text(판매자수, prev_판매자수)}
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div style="background-color:#F3FBF5; padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
            <div style="font-size:16px;">방송 수</div>
            <div style="font-size:28px; font-weight:bold;">{방송수:,}</div>
            {get_delta_text(방송수, prev_방송수)}
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()
    
    # ===== 시간대별 방송 실적 =========
    st.markdown("""
    <div style='background-color:#DFF5E1; padding:10px 15px; border-radius:8px; margin-bottom:10px;'>
    <span style='font-weight:bold; font-size:22px;'>⏰ 시간대별 방송 실적 비교</span>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    
    metrics = ["총 매출액(원)", "방송조회수", "구매 전환율"]
    agg_funcs = {
        '총 매출액(원)': 'sum',
        '방송조회수': 'sum',
        '구매 전환율': 'mean'
    }

    hourly = df_b.groupby('방송시').agg(agg_funcs).reset_index()

    cols = st.columns(3)

    for i, metric in enumerate(metrics):
        max_val = hourly[metric].max()
        min_val = hourly[metric].min()

        fig = px.bar(
            hourly,
            x='방송시',
            y=metric,
            title=metric
        )
        fig.update_traces(marker_color="#4CAF50", textposition='outside')
        fig.update_layout(margin=dict(t=30, b=20, l=10, r=10), height=280)

        with cols[i]:
            st.plotly_chart(fig, use_container_width=True)
            
    st.divider()
    
    # ====== 인구 통계 분포 =====
    st.markdown("""
    <div style='background-color:#DFF5E1; padding:10px 15px; border-radius:8px; margin-bottom:10px;'>
    <span style='font-weight:bold; font-size:22px;'>👥 인구 통계 분포</span>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 성별 타겟 분포")

        gender_target_counts = df_b['성별 타겟군'].dropna().str.strip().value_counts().reset_index()
        gender_target_counts.columns = ['성별 타겟군', '건수']

        fig_donut = px.pie(
            gender_target_counts,
            names='성별 타겟군',
            values='건수',
            hole=0.5,
            color_discrete_sequence=['#C8E6C9', '#81C784', '#388E3C']  # 여성, 균형, 남성 (예시)
        )
        fig_donut.update_traces(textinfo='percent+label', textposition='inside')

        fig_donut.update_layout(
            height=270,
            margin=dict(t=20, b=10, l=0, r=0),
            showlegend=True
        )

        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        st.markdown("##### 연령 타겟 분포")

        age_cols = ['10대', '20대', '30대', '40대', '50대', '60대']
        age_avg = df_b[age_cols].mean().reset_index()
        age_avg.columns = ['연령대', '비율']

        fig_age = px.bar(
            age_avg,
            x='연령대',
            y='비율',
            text='비율',
            labels={'비율': '비율 (%)'},
            title=None,
            color='연령대',
                color_discrete_sequence=[
        '#E8F5E9',  # 1: 매우 연함
        '#C8E6C9',  # 2
        '#A5D6A7',  # 3
        '#81C784',  # 4
        '#66BB6A',  # 5
        '#388E3C'   # 6: 진함
    ]
        )
        fig_age.update_traces(texttemplate='%{text:.1f}%', textposition='outside')

        fig_age.update_layout(
            yaxis_range=[0, 45],
            margin=dict(t=20, b=10, l=10, r=10),
            height=270,
            showlegend=False
        )

        st.plotly_chart(fig_age, use_container_width=True)
        
    st.divider()
            
    # ======= 주요 카테고리 성과 비교  ========
    st.markdown("""
    <div style='background-color:#DFF5E1; padding:10px 15px; border-radius:8px; margin-bottom:10px;'>
    <span style='font-weight:bold; font-size:22px;'>📌 주요 카테고리 성과 비교</span>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 지표별 상위 카테고리")

        top_sales = df_b.groupby('대분류')['총 매출액(원)'].sum().sort_values(ascending=False).head(3).index.tolist()
        top_volume = df_b.groupby('대분류')['총 판매량'].sum().sort_values(ascending=False).head(3).index.tolist()
        top_views = df_b.groupby('대분류')['방송조회수'].sum().sort_values(ascending=False).head(3).index.tolist()
        
        summary_df = pd.DataFrame({
            "총 매출액 상위": top_sales,
            "총 판매량 상위": top_volume,
            "방송조회수 상위": top_views
        })

        st.dataframe(summary_df.style.hide(axis="index"), use_container_width=True) 

    with col2:
        st.markdown("##### 구매 전환율 상위 카테고리")

        conv = (
            df_b.groupby('대분류')['구매 전환율']
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
        )

        fig = px.bar(
            conv,
            x='대분류',
            y='구매 전환율',
            title='',
            text_auto='.2f'
        )
        fig.update_traces(marker_color="#4CAF50")
        fig.update_layout(
            xaxis={'categoryorder':'total descending'},
            margin=dict(t=20, b=20, l=10, r=10),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
    st.divider()

    # =======매출/판매 통계==========
    st.markdown("""
    <div style='background-color:#DFF5E1; padding:10px 15px; border-radius:8px; margin-bottom:10px;'>
    <span style='font-weight:bold; font-size:22px;'>📈 일별 매출 및 판매 추이</span>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    # 일별 집계
    by_day = df_b.groupby('방송일').agg({
        '총 매출액(원)': 'sum',
        '총 판매량': 'sum'
    }).reset_index()

    col1, col2 = st.columns(2)

    with col1:
        chart_df = by_day.rename(columns={'총 매출액(원)': '매출금액'})
        max_val = chart_df['매출금액'].max()
        min_val = chart_df['매출금액'].min()

        def label_func(val):
            if val == max_val or val == min_val:
                return f"{val:,.0f}"
            else:
                return ""

        chart_df['라벨'] = chart_df['매출금액'].apply(label_func)

        fig = px.line(
            chart_df,
            x='방송일',
            y='매출금액',
            text='라벨',
            markers=True
        )
        fig.update_traces(line_color= '#429845', textposition="top center")
        fig.update_layout(
            title='총 매출액(원)',
            margin=dict(t=30, b=20, l=10, r=10),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        chart_df = by_day.rename(columns={'총 판매량': '판매수량'})
        max_val = chart_df['판매수량'].max()
        min_val = chart_df['판매수량'].min()

        chart_df['라벨'] = chart_df['판매수량'].apply(label_func)

        fig = px.line(
            chart_df,
            x='방송일',
            y='판매수량',
            text='라벨',
            markers=True
        )
        fig.update_traces(line_color= '#429845', textposition="top center")
        fig.update_layout(
            title='총 판매량',
            margin=dict(t=30, b=20, l=10, r=10),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()

# Plus 대시보드 page
elif st.session_state.page == 'Plus':

    st.title('Plus 대시보드')
    st.caption('전략 수립에 필요한 실전형 분석 도구로, 성과를 높일 수 있는 방향을 제시합니다.')
    
    # ======== [공통 필터 : 대분류 선택] ========== #
    all_main_cats = plus_t['대분류'].dropna().unique()
    selected_main_cat = st.selectbox("분석할 카테고리를 선택하세요", sorted(all_main_cats))

    p_t = plus_t[plus_t['대분류'] == selected_main_cat]
    df_ad_cat = df_ad[df_ad['대분류'] == selected_main_cat]
    df_price = open_df[open_df['대분류'] == selected_main_cat]
    
    # ======== 시간대 분석 ========== #
    st.markdown("""
    <div style='background-color:#DFF5E1; padding:10px 15px; border-radius:8px; margin-bottom:10px;'>
    <span style='font-weight:bold; font-size:22px;'>🎯 최적의 방송 시간대 분석</span>
    </div>
    """, unsafe_allow_html=True)
    
    planned_comment_dict = {
        "가구/인테리어" : "기획 방송은 10-11시와 18-20시에 평균 이상의 방송 수가 집중되어 있으며, 특히 19시는 평균의 4배를 넘는 방송 수로 방송 밀집도가 높게 나타났습니다.",
        "도서" : "10-12시와 18시는 평균 대비 2배 이상의 방송이 집중되어, 해당 시간대에 과밀화된 편성 경향이 나타났습니다.",
        "디지털/가전" : "오전 10-11시와 18-19시에 방송 편성이 밀집되며 경쟁이 집중되는 양상을 보였습니다.",
        "생활/건강" : "기획전은 오전 10-11시와 18-20시에 방송이 집중되었으며, 특히 10-11시는 평균 대비 3~4배로 매우 높게 집계되었습니다.",
        "스포츠/레저" : "기획 방송은 18-19시에 집중적으로 편성되었으며, 12-16시대에는 방송 수가 평균 이하로 낮게 나타났습니다.",
        "식품" : "기획 방송은 오전 9-11시에 집중적으로 편성되었으며, 특히 10시에 방송 수가 가장 많았습니다.",
        "여가/생활편의" : "기획 방송은 18-19시에 집중 편성되며 이 시간대에 경쟁이 치열하게 나타났습니다.",
        "출산/육아" : "기획 방송은 오전 10-11시에 집중되어 편성되었으며, 12시 이후부터는 방송 수가 급격히 감소했습니다.",
        "패션의류" : "기획 방송은 18-20시에 집중적으로 편성되었으며, 특히 19시에 방송 수가 가장 많았습니다.",
        "패션잡화" : "기획 방송은 10-11시와 18-20시에 집중적으로 편성되었으며, 특히 19시에 방송 수가 최고치를 기록했습니다.",
        "화장품/미용" : "기획 방송은 오전 10시와 18-19시에 집중되었으며, 12-17시대에는 방송 편성이 현저히 적게 나타났습니다."
        
    }

    top_hour_comment_dict = {
        "가구/인테리어": "오픈 방송의 상위 매출 달 분석 결과, 20시에 가장 높은 평균 매출액을 기록했습니다.",
        "도서" : "오픈 방송의 상위 매출 달 분석 결과, 10-12시에 총 매출이 집중되었고 특히 12시에 최고 매출을 기록했습니다.",
        "디지털/가전" : "오픈 방송은 11시, 16시, 22시에 주요 매출 피크가 형성되었고, 이 시간대에 성과가 집중되는 경향이 나타났습니다.",
        "생활/건강" : "오픈 방송의 평균 총매출은 20시에 가장 높았고, 반면 12시와 21시 이후 시간대는 매출 성과가 저조하게 나타났습니다.",
        "스포츠/레저" : "오픈 방송은 10시, 14시, 17시에 평균 매출이 가장 높아 주요 성과 시간대로 확인되었습니다.",
        "식품" : "오픈 방송은 9시에 총매출이 압도적으로 높게 나타났고, 두 방송 유형 모두 오전 시간대에 성과가 집중되는 경향을 보였습니다.",
        "여가/생활편의" : "오픈 방송은 11시에 압도적으로 높은 매출 성과를 기록해 성과 집중 구간으로 확인되었습니다.",
        "출산/육아" : "오픈 방송은 오전 10시에 가장 높은 매출 성과를 기록했으며, 8시와 19시대에도 꾸준한 매출 흐름이 관찰되었습니다.",
        "패션의류" : "오픈 방송은 19시에 상위 매출 피크를 기록했으며, 기획 방송의 방송 수와는 크게 연동되지 않는 독립적인 성과 흐름을 보였습니다.",
        "패션잡화" : "오픈 방송은 18시에 매출이 급격히 상승하며 피크를 형성했고, 이 성과가 19시까지 이어졌습니다. 반면 15-16시대는 활동성이 낮아 성과가 저조했습니다.",
        "화장품/미용" : "오픈 방송은 18시에 최고 매출을 기록했고, 19시에도 높은 성과가 이어졌습니다. 12-16시대는 매출 데이터가 부족해 활동이 저조한 양상을 보였습니다."
        
    }

    summary_comment_dict = {
        "가구/인테리어": "19-20시는 기획 방송에 유입이 몰릴 가능성이 높으므로, 오픈 방송은 17-18시에 선제적으로 편성하거나 20시 이후로 이동해 콘텐츠 경쟁력을 극대화하는 전략을 권장드립니다.",
        "도서" : "⇒ 기획 방송이 몰리는 10-12시는 경쟁이 치열하지만, 오픈 방송의 성과도 뚜렷하게 나타나는 시간대이므로 단순 회피보다는 전략적 접근이 필요합니다. 동일 시간대에 기획 방송과 차별화된 콘텐츠나 소분류 상품을 활용해 경쟁력을 확보하고 상위 매출을 달성하는 방안을 권장드립니다.",
        "디지털/가전" : "⇒ 경쟁 강도가 높은 10-11시와 18-19시대에는 차별화된 전략으로 경쟁력을 확보해 11시 성과 피크를 놓치지 않도록 하세요. 또한 경쟁이 상대적으로 적은 16시와 22시대를 활용해 안정적인 시청자 유입과 매출 극대화를 동시에 달성하는 전략을 권장드립니다.",
        "생활/건강" : "⇒ 오전 10-11시와 20시대를 성과 집중 구간으로 설정해 적극 공략하고, 18-19시대부터 고객 유입을 선제적으로 유도해 20시 매출 피크로 자연스럽게 연결될 수 있도록 방송 편성을 설계하세요. 한편, 매출 효율성이 낮은 12시와 21시 이후 시간대는 리소스 투입을 최소화하는 전략을 권장드립니다.",
        "스포츠/레저" : "⇒ 경쟁이 비교적 적은 14-17시대를 중심으로 단독 방송을 편성해 틈새 시장을 효과적으로 공략하고, 안정적인 시청자 확보와 함께 매출 성과를 극대화하는 전략을 권장드립니다.",
        "식품" : "⇒ 오전 9-11시대를 핵심 공략 시간대로 활용하고, 경쟁이 덜한 8시대나 11시 이후에 테스트 방송을 시도해 시청자 반응을 확인한 뒤 성과 가능성을 탐색하고 확장하는 전략을 권장드립니다.",
        "여가/생활편의" : "⇒ 경쟁 강도가 높은 18-19시대를 피해, 성과가 집중된 10-11시대에 방송을 편성하여 안정적인 시청자 유입과 매출 극대화를 동시에 달성하는 전략을 권장드립니다.",
        "출산/육아" : "⇒ 기획과 오픈 방송 모두 성과가 집중되는 10-11시대를 핵심 공략 시간대로 삼고, 차별화된 콘텐츠를 통해 경쟁력을 강화하세요. 동시에 8시와 19시대에 분산 편성을 시도해 안정적인 시청자 유입과 성과 확대를 도모하는 전략을 권장드립니다.",
        "패션의류" : "⇒ 오픈 방송의 성과가 기획 방송과 큰 영향을 주고받지 않는 특성을 활용해, 경쟁이 비교적 적은 9시, 12시, 14시대를 집중 공략하여 안정적인 실적을 확보하고 성과를 극대화하는 전략을 권장드립니다.",
        "패션잡화" : "⇒ 기획과 오픈 방송 모두 18-19시대는 핵심 공략 시간대로, 경쟁이 심해지기 전 선제적으로 방송을 편성해 시청자를 확보해보세요. 또한 활동성이 낮은 15-16시는 피하고, 10-11시대를 보조 전략으로 활용해 안정적인 실적을 추구하는 방안을 권장드립니다.",
        "화장품/미용" : "⇒ 기획 방송과 오픈 방송 모두 활발한 성과를 보인 18-19시대를 핵심 공략 시간대로 삼고, 오전 9-11시대를 보조 전략으로 활용하세요. 또한 17시에 테스트 방송을 운영해 시청자 반응을 파악하고, 긍정적인 결과가 확인되면 정규 편성으로 확장하는 방안을 권장드립니다."
    }

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('##### 기획 방송 집중 시간대')
 
        st.caption("✅ 이 시간대는 피해서 방송을 편성하면, 비교적 안정적인 유입을 확보할 수 있습니다.")
        
        df_planned = p_t[p_t['유형'] == '기획']

        hour_counts_planned = df_planned['방송시'].value_counts().sort_index().reset_index()
        hour_counts_planned.columns = ['방송시', '기획 방송 수']
        mean_count = hour_counts_planned['기획 방송 수'].mean()

        fig1 = px.bar(
            hour_counts_planned,
            x='방송시',
            y='기획 방송 수'
        )
        
        fig1.update_traces(marker_color="#4CAF50")
        fig1.add_hline(
            y=mean_count,
            line_dash="dot",
            line_color="red",
            annotation_text=f"평균",
            annotation_position="top left"
        )
    
        fig1.update_layout(
            height=300,
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        with st.expander("🔍 기획 방송 해석 보기"):
            if selected_main_cat in planned_comment_dict:
                st.markdown(planned_comment_dict[selected_main_cat])

    with col2:
        st.markdown('##### 성과가 높았던 시간대')
 
        st.caption("✅ 상위 30% 셀러는 이 시간대를 공략했습니다. 해당 시간대를 중심으로 테스트해 보세요.")

        df_open_top = p_t[(p_t['유형'] == '오픈') & (p_t['매출상위'] == '상위30%')]
                
        hour_sales_open_top = df_open_top.groupby('방송시')['총 매출액(원)'].mean().reset_index()
        
        top_hour = hour_sales_open_top.sort_values(by='총 매출액(원)', ascending=False).iloc[0]['방송시']

        hour_sales_open_top['강조'] = hour_sales_open_top['방송시'].apply(
            lambda x: '강조' if x == top_hour else '기본'
        )

        fig2 = px.bar(
            hour_sales_open_top,
            x='방송시',
            y='총 매출액(원)',
            color='강조',
            color_discrete_map={
                '강조': '#2E7D32', 
                '기본': '#4CAF50'  
            }
        )
        
        fig2.update_layout(
            height=300,
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        with st.expander("🔍 오픈 방송 해석 보기"):
            if selected_main_cat in top_hour_comment_dict:
                st.markdown(top_hour_comment_dict[selected_main_cat])
            
    if selected_main_cat in summary_comment_dict:
        st.markdown(
            f"""
            <div style="background-color:#f9f9f9; border-left: 6px solid #2E7D32; padding: 1rem 1.2rem; border-radius: 12px; margin-top: 1.5rem;">
                <div style="font-weight:bold; font-size:1.1rem; margin-bottom:0.3rem;">💭 최종 전략 가이드</div>
                <div style="font-size:0.95rem; color:#333; line-height:1.6;">
                    {summary_comment_dict[selected_main_cat]}
                </div>
                <div style="font-size:0.8rem; color:gray; margin-top:0.6rem;">
                    👉 더 자세한 해석은 위의 <b>기획 방송 해석</b>과 <b>오픈 방송 해석</b>을 확인해 보세요.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        
    st.markdown("---")
    
    # ===== 가격 구간별 효과 분석 - 버블 차트 ======== 
    
    st.markdown("""
    <div style='background-color:#DFF5E1; padding:10px 15px; border-radius:8px; margin-bottom:10px;'>
    <span style='font-weight:bold; font-size:22px;'>💰 가격 구간별 효과 분석</span>
    </div>
    """, unsafe_allow_html=True)
    st.caption("가격대별 성과 및 효과 크기를 통해 최적의 판매 가격 전략을 도출할 수 있습니다.")
    
    color_map = {
        1: 'black',
        2: 'skyblue',
        3: 'lightgreen',
        4: 'green'
    }
    def get_emoji(label):
        if label == 'opportunity':
            return '💡'
        elif label == 'test':
            return '✅'
        else:
            return ''

    agg = (
        df_price
        .groupby('가격 구간', as_index=False)
        .agg(
            효과크기=('효과 크기', 'mean'),
            평균판매량=('1회 방송당 판매량', 'mean'),
            라벨링=('라벨링', 'first')
        )
        .dropna()
        .sort_values('가격 구간')
    )

    agg['버블크기'] = agg['효과크기'] ** 3 * 30
    agg['버블색'] = agg['효과크기'].round().map(color_map)
    agg['hover'] = "라벨링: " + agg['라벨링'].astype(str)
    agg['라벨텍스트'] = agg['라벨링'].apply(get_emoji)

    fig = px.scatter(
        agg,
        x='가격 구간',
        y='평균판매량',
        size='버블크기',
        color='효과크기',
        text = '라벨텍스트',
        color_continuous_scale=['black', 'skyblue', 'lightgreen', 'green'],
        hover_name='hover',
        size_max=60
    )

    fig.update_traces(
    textposition='top center',
    textfont=dict(
        size=12,
        color='black',
        family='Arial'
    ),
    marker=dict(
        line=dict(width=0)
    )
)

    fig.update_layout(
        height=450,
        margin=dict(t=40, b=40, l=30, r=30),
        xaxis_title="가격 구간",
        yaxis_title="1회 방송당 판매량",
        showlegend=True,
        coloraxis_showscale=False
    )

    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📘 이모티콘 해석 가이드"):
        st.markdown("""
        - 💡 **기회 구간**  
        기획방송에서 부진하거나 미판매하는 구간입니다. 오픈라방만의 경쟁력을 강화해보세요!

        - ✅ **테스트 권장 구간**  
        타 구간 대비 **높은 효과가 검증**되었으나 테스트 방송이 필요한 구간입니다. 파일럿 방송을 통해 성과에 따라 추후 편성 여부를 결정하세요!
        """)
    
    st.markdown("---")
    
    # ===== 상품 클러스터링 ==========

    card_dict = {i: f"card/cluster_{i}.png" for i in range(8)}

    cluster_name_dict = {
        0: '성과 안정 밸런스형',
        1: '성과 최하위형',
        2: '알림고객 관리 필요형',
        3: '성과 저조형',
        4: '전환 대비 매출 약세형',
        5: '유입 대비 전환 약세형',
        6: '유입 대비 판매 효율형',
        7: '매출 최상위형'
    }

    # 클러스터별 날씨 민감도 점수
    weather_score = {
        0: 0, 1: 0, 2: 0,
        3: 2,            
        4: 1, 5: 1, 6: 1, 7: 1 
    }
    st.markdown("""
    <div style='background-color:#DFF5E1; padding:10px 15px; border-radius:8px; margin-bottom:10px;'>
    <span style='font-weight:bold; font-size:22px;'>💡 상품 전략 추천 시스템</span>
    </div>
    """, unsafe_allow_html=True)

    categories = ["선택하세요"] + sorted(cluster_df["대분류"].unique())
    genders = ["선택하세요"] + list(cluster_df["성별 타겟군"].unique())

    col1, col2 = st.columns([1,2])
    with col1:
        
        selected_cat = st.selectbox("카테고리 선택", categories, index=0)
        
        selected_gender = st.selectbox("성별 타겟 선택", genders, index=0)
        
        weather_sensitive = st.radio("📡 평소 날씨 변화에 따라 판매 성과가 크게 달라지셨나요?", ["예", "아니오"], index=0)
    with col2:
        # ===== 필터 조건이 모두 선택된 경우 =====
        if selected_cat != "선택하세요" and selected_gender != "선택하세요" and weather_sensitive != "선택 안 함":

            selected_cluster = None
            reason = ""

            # 대분류, 성별 타겟군 각각 필터링
            cat_match = cluster_df[cluster_df["대분류"] == selected_cat]
            gender_match = cluster_df[cluster_df["성별 타겟군"] == selected_gender]

            # 교집합 추출
            intersection = pd.merge(cat_match, gender_match, how='inner')

            if not intersection.empty:
                cluster_candidates = intersection["cluster"]

                if weather_sensitive == "예":
                    if 3 in cluster_candidates.values:
                        selected_cluster = 3
                        #reason = "✅ 교집합에 클러스터 3이 포함되고, 날씨 영향이 '예'로 선택되어 클러스터 3을 추천합니다."
                    else:
                        # 점수 최고 → 그 중 빈도수 최다
                        scored = intersection.copy()
                        scored["점수"] = scored["cluster"].map(weather_score)
                        max_score = scored["점수"].max()
                        top_group = scored[scored["점수"] == max_score]
                        selected_cluster = top_group["cluster"].value_counts().idxmax()
                        #reason = f"✅ 교집합에 클러스터 3은 없지만, 날씨 민감도 점수 {max_score}점인 클러스터 중 가장 빈도 높은 클러스터를 추천합니다."
                else:
                    # 날씨 영향 없음일 경우 → 점수 낮은 순 → 빈도수
                    scored = intersection.copy()
                    scored["점수"] = scored["cluster"].map(weather_score)
                    min_score = scored["점수"].min()
                    top_group = scored[scored["점수"] == min_score]
                    selected_cluster = top_group["cluster"].value_counts().idxmax()
                    #reason = f"✅ 날씨 영향이 '아니오'로 선택되어, 날씨 민감도 점수 {min_score}점인 클러스터 중 가장 빈도 높은 클러스터를 추천합니다."

            else:
                # 교집합 없음 → 대분류 기준
                cat_only = cat_match.copy()

                if weather_sensitive == "예":
                    if 3 in cat_only["cluster"].values:
                        selected_cluster = 3
                        #reason = "✅ 대분류 기준 클러스터 중 클러스터 3이 존재하고, 날씨 영향이 '예'로 선택되어 클러스터 3을 추천합니다."
                    else:
                        scored = cat_only.copy()
                        scored["점수"] = scored["cluster"].map(weather_score)
                        max_score = scored["점수"].max()
                        top_group = scored[scored["점수"] == max_score]
                        selected_cluster = top_group["cluster"].value_counts().idxmax()
                        #reason = f"✅ 날씨 민감도 점수 {max_score}점인 클러스터 중 대분류 기준으로 가장 빈도 높은 클러스터를 추천합니다."
                else:
                    scored = cat_only.copy()
                    scored["점수"] = scored["cluster"].map(weather_score)
                    min_score = scored["점수"].min()
                    top_group = scored[scored["점수"] == min_score]
                    selected_cluster = top_group["cluster"].value_counts().idxmax()
                    #reason = f"✅ 날씨 영향이 '아니오'로 선택되어, 날씨 민감도 점수 {min_score}점인 클러스터 중 대분류 기준으로 가장 빈도 높은 클러스터를 추천합니다."

            # 추천 결과 출력
            seller_name = st.session_state.get('seller_name', '')
            
            if selected_cluster is not None:
                selected_cluster_name = cluster_name_dict[selected_cluster]
                st.markdown(f"#### 🎯 **{seller_name}**님의 판매 상품은 **{selected_cluster_name}** 에 해당합니다.")

                st.image(card_dict[selected_cluster], use_container_width = True)
                st.caption("📌 이 카드는 클러스터 기반 전략을 요약한 콘텐츠입니다.")
        else:
            st.success("카테고리, 성별 타겟, 날씨 민감도를 모두 선택해주세요.")

    st.divider()
    # ======= 광고 분석 ==========
    st.markdown("""
    <div style='background-color:#DFF5E1; padding:10px 15px; border-radius:8px; margin-bottom:10px;'>
    <span style='font-weight:bold; font-size:22px;'>📢 광고 성과 분석</span>
    </div>
    """, unsafe_allow_html=True)

    st.caption("광고 전략 수립 시, 카테고리별 성별 타겟군의 특성과 각 광고상품의 성과를 함께 고려해 보세요.")
    
    gender_groups = df_ad_cat['성별 타겟군'].dropna().unique()
    cols = st.columns(len(gender_groups))

    for i, gender in enumerate(gender_groups):
        with cols[i]:
            st.markdown(f"**{gender}**")
            subset = df_ad_cat[df_ad_cat['성별 타겟군'] == gender].sort_values(by='광고상품 ROI', ascending=False).head(5)

            fig = px.bar(
                subset,
                x='광고상품 ROI',
                y='광고상품 리스트',
                orientation='h',
                text='광고상품 ROI'
            )
            fig.update_traces(marker_color="#4CAF50", texttemplate='%{text:.1f}', textposition='outside')
            
            if len(subset) == 1:
                fig.update_layout(height=100)
            elif len(subset) == 2:
                fig.update_layout(height=200)
            else:
                fig.update_layout(height=300)
            
            fig.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                yaxis=dict(categoryorder='total ascending')
            )
            st.plotly_chart(fig, use_container_width=True)
            
    st.divider()       
#########################################
# Pro 대시보드 page
elif st.session_state.page == 'Pro':
    
    st.title('Pro 대시보드')
    st.caption('셀러 개인별 맞춤형 피드백을 제공하고 방송 전략을 제안합니다.')
    st.markdown(f"""
    <div style='background-color:#DFF5E1; padding:10px 15px; border-radius:8px; margin-bottom:10px;'>
    <span style='font-weight:bold; font-size:22px;'>📋 {st.session_state.seller_name} 셀러의 방송 기록</span>
    </div>
    """, unsafe_allow_html=True)
    
    seller_df = df_pro[df_pro['스토어명'] == st.session_state.seller_name].reset_index(drop=True)
    
    # 빅넘버 메트릭
    met_col1, met_col2, met_col3, met_col4 = st.columns(4)
    
    방송_수 = seller_df.shape[0]
    총_매출 = int(seller_df['총 매출액(원)'].mean())
    평균_조회수 = int(seller_df['방송조회수'].mean())
    판매량 = seller_df['총 판매량'].mean().round(2)
    
    with met_col1:
        st.markdown(f"""
        <div style="background-color:#F3FBF5; padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
            <div style="font-size:16px;">전체 방송 수</div>
            <div style="font-size:28px; font-weight:bold;">{방송_수}개</div>
            
        </div>
        """, unsafe_allow_html=True)
                
    with met_col2:
        st.markdown(f"""
        <div style="background-color:#F3FBF5; padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
            <div style="font-size:16px;">평균 조회수</div>
            <div style="font-size:28px; font-weight:bold;">{평균_조회수}회</div>
            
        </div>
        """, unsafe_allow_html=True)
        
    with met_col3:
        st.markdown(f"""
        <div style="background-color:#F3FBF5; padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
            <div style="font-size:16px;">평균 판매량</div>
            <div style="font-size:28px; font-weight:bold;">{판매량:,}개</div>
            
        </div>
        """, unsafe_allow_html=True)
                       
    with met_col4:
        st.markdown(f"""
        <div style="background-color:#F3FBF5; padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
            <div style="font-size:16px;">평균 매출</div>
            <div style="font-size:28px; font-weight:bold;">{총_매출:,}원</div>
            
        </div>
        """, unsafe_allow_html=True)
        

    # 상위 10개 정렬 기준 
    seller_df_sorted = seller_df.sort_values(by='시작시간', ascending=False).head(30)

    # 선 그래프
    line_col1, line_col2, line_col3 = st.columns(3)
    
    with line_col1:
        fig = px.line(
        seller_df_sorted,
        x='시작시간',
        y='방송조회수',
        markers=True,
        title='최근 방송 조회수 추이',
        labels={'시작시간': '시작시간', '방송조회수': '조회수'}
        )
        fig.update_traces(line_color="#429845")
        st.plotly_chart(fig, use_container_width=True)
       
    with line_col2:
        fig = px.line(
        seller_df_sorted,
        x='시작시간',
        y='구매 전환율',
        markers=True,
        title='최근 방송 전환율 추이',
        labels={'시작시간': '시작시간', '구매 전환율': '전환율'}
        )
        fig.update_traces(line_color="#429845")
        st.plotly_chart(fig, use_container_width=True)
        
    with line_col3:
        fig = px.line(
        seller_df_sorted,
        x='시작시간',
        y='총 매출액(원)',
        markers=True,
        title='최근 방송 매출 추이',
        labels={'시작시간': '시작시간', '총 매출액(원)': '매출액'}
        )
        fig.update_traces(line_color="#429845")
        st.plotly_chart(fig, use_container_width=True)
        

    # 데이터프레임 해당 셀러의 정보 데이터 프레임
    st.dataframe(seller_df)
    
    st.divider()
    #######################################################
    # 2. 방송 클러스터링 결과 확인 & 맞춤 행동 지침 제공
        
    # 👉 방송 유형별 인사이트 제공
    broadcast_type = [
        '노출 실패형',
        '전환 실패형',
        '잠재 성장형',
        '보통 방송',
        '조회 편중형',
        '전환 집중형',
        '최다 매출형'
        ]

    wide1, wide2 = st.columns(2)
    # 방송 클러스터링 결과
    
    with wide1:
        st.markdown(f"""
    <div style='background-color:#DFF5E1; padding:10px 15px; border-radius:8px; margin-bottom:10px;'>
    <span style='font-weight:bold; font-size:22px;'>📺 방송 유형 분석</span>
    </div>
    """, unsafe_allow_html=True)
        
        cluster_option = st.selectbox(
        '방송 유형을 선택하세요',
        options=broadcast_type,
        index=0
    )
        
        if cluster_option == '노출 실패형':
            st.markdown(f'#### 1️⃣ {cluster_option}')
            st.error('##### 노출 자체에 실패한 방송')
            with st.container():
                col1, col2, col3 = st.columns(3)
                col1.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">평균 지표</div>
                <div style="font-size:20px; font-weight:bold;">최하위</div>
            </div>
            """, unsafe_allow_html=True)
                
                col2.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">방송 비율</div>
                <div style="font-size:20px; font-weight:bold;">19.58%</div>
            </div>
            """, unsafe_allow_html=True)

                col3.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">평균 매출</div>
                <div style="font-size:20px; font-weight:bold;">0.19원</div>
            </div>
            """, unsafe_allow_html=True)

                col4, col5, col6 = st.columns(3)
                col4.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">분당 유입</div>
                <div style="font-size:20px; font-weight:bold;">1.88명</div>
            </div>
            """, unsafe_allow_html=True)

                col5.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">구매 고객</div>
                <div style="font-size:20px; font-weight:bold;">0.13명</div>
            </div>
            """, unsafe_allow_html=True)
                
                col6.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">실유입 대비 판매율</div>
                <div style="font-size:20px; font-weight:bold;">0.06%</div>
            </div>
            """, unsafe_allow_html=True)
                
            st.write('')
            
            st.markdown(f'''         
                        ##### 🔍 원인 진단
                        - 썸네일/제목의 매력 부족으로 클릭 유도 실패
                        - 방송 시간대의 전략적 미스
                        - 방송 초반 후킹 부족으로 이탈률 상승
                        
                        ##### 전략 방향 ➡️ 첫 진입 장벽 해소
                        ''')
            st.error('''
                        - 썸네일 인상적으로 구성
                        - 방송 초반 시각적 요소 및 멘트로 강한 후킹
                        - **꾸준한 방송 진행**
                     ''')
            
        elif cluster_option == '전환 실패형':
            st.markdown(f'#### 2️⃣ {cluster_option}')
            st.error('##### 노출엔 성공했지만 구매 전환에 실패한 방송')
            with st.container():
                
                col1, col2, col3 = st.columns(3)
                col1.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">평균 지표</div>
                <div style="font-size:20px; font-weight:bold;">구매 없음</div>
            </div>
            """, unsafe_allow_html=True)
                
                col2.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">방송 비율</div>
                <div style="font-size:20px; font-weight:bold;">7.45%</div>
            </div>
            """, unsafe_allow_html=True)

                col3.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">평균 매출</div>
                <div style="font-size:20px; font-weight:bold;">0.02원</div>
            </div>
            """, unsafe_allow_html=True)
                
                col4, col5, col6 = st.columns(3)
                col4.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">분당 유입</div>
                <div style="font-size:20px; font-weight:bold;">33.01명</div>
            </div>
            """, unsafe_allow_html=True)

                col5.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">구매 고객</div>
                <div style="font-size:20px; font-weight:bold;">1.44명</div>
            </div>
            """, unsafe_allow_html=True)
                
                col6.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">실유입 대비 판매율</div>
                <div style="font-size:20px; font-weight:bold;">0.01%</div>
            </div>
            """, unsafe_allow_html=True)
                
            st.write('')
                
            st.markdown(f'''
                        ##### 🔍 원인 진단
                        - 제품 매력 부족
                        - 제품에 대한 신뢰 부족
                        - 구매 망설임 요소 다수 존재
                        
                        ##### 전략 방향 ➡️ 상품 개선
                        ''')
            st.error('''
                        - **상품 정보 핵심 포인트**로 3가지 이점만 정리해 간략히 제시
                        - 후기 별점 사용기 등 사회적 증거 요소 강하게 어필
                        - **왜 지금 사야하는가?** 어필 → 긴박감있는 멘트/자막
                     ''')
            
        elif cluster_option == '잠재 성장형':
            st.markdown(f'#### 3️⃣ {cluster_option}')
            st.warning('##### 유입/전환/매출 모두 평균 이하인 방송')
            with st.container():
                col1, col2, col3 = st.columns(3)
                col1.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">평균 지표</div>
                <div style="font-size:20px; font-weight:bold;">하위권</div>
            </div>
            """, unsafe_allow_html=True)
                
                col2.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">방송 비율</div>
                <div style="font-size:20px; font-weight:bold;">20.62%</div>
            </div>
            """, unsafe_allow_html=True)

                col3.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">평균 매출</div>
                <div style="font-size:20px; font-weight:bold;">161,809원</div>
            </div>
            """, unsafe_allow_html=True)
                
                col4, col5, col6 = st.columns(3)
                col4.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">분당 유입</div>
                <div style="font-size:20px; font-weight:bold;">2.97명</div>
            </div>
            """, unsafe_allow_html=True)

                col5.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">구매 고객</div>
                <div style="font-size:20px; font-weight:bold;">2.35명</div>
            </div>
            """, unsafe_allow_html=True)
                
                col6.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">실유입 대비 판매율</div>
                <div style="font-size:20px; font-weight:bold;">1.32%</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write('')
            
            st.markdown(f'''
                        ##### 🔍 원인 진단
                        - **유입/전환/매출 모두 평균 이하**
                        - 제품 메시지 혼선 및 강점이 명확하지 않음
                        - 오프닝에서 시청 이유 전달 실패

                        ##### 전략 방향 ➡️ 기본기 강화 + 핵심 포인트 명확화
                        ''')
            
            st.warning('''                  
                        - 방송 오프닝 구조 변경 → 시청 이유를 10초 내에 제시
                        - 제품 소개 목적 명확화 (정보 전달 vs 혜택 강조 등)
                        - 고객 구매 여정을 시각적으로 흐름화
                        - “궁금한 점 있으시면 댓글로 남겨주세요” 등 리액션 유도 멘트 삽입
                        - 단일 목적 콘텐츠(정보형/체험형/이벤트형 등) 실험
                        ''')   
            
        elif cluster_option == '보통 방송':
            st.markdown(f'#### 4️⃣ {cluster_option}')
            st.warning('##### 전체적으로 나쁘지 않은 방송')
            with st.container():
                col1, col2, col3 = st.columns(3)
                col1.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">평균 지표</div>
                <div style="font-size:20px; font-weight:bold;">중위권</div>
            </div>
            """, unsafe_allow_html=True)
                
                col2.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">방송 비율</div>
                <div style="font-size:20px; font-weight:bold;">17.49%</div>
            </div>
            """, unsafe_allow_html=True)

                col3.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">평균 매출</div>
                <div style="font-size:20px; font-weight:bold;">2,040,942원</div>
            </div>
            """, unsafe_allow_html=True)
                
                col4, col5, col6 = st.columns(3)
                col4.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">분당 유입</div>
                <div style="font-size:20px; font-weight:bold;">16.33명</div>
            </div>
            """, unsafe_allow_html=True)

                col5.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">구매 고객</div>
                <div style="font-size:20px; font-weight:bold;">15.03명</div>
            </div>
            """, unsafe_allow_html=True)
                
                col6.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">실유입 대비 판매율</div>
                <div style="font-size:20px; font-weight:bold;">1.34%</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write('')
                
            st.markdown(f'''
                        ##### 🔍 원인 진단
                        - 전체적으로 무난한 지표
                        - **명확한 강점 부재** → 정체 가능성 있음

                        ##### 전략 방향 ➡️ 실험 기반 최적화
                        ''')
            
            st.warning('''                  
                        - 상품 소개 순서, 리뷰 강조 등 **A/B 테스트 진행**
                        - 클러스터 3·4 요소 혼합 → 후킹/리뷰 강조
                        - 정보형/예능형/Q&A형 콘텐츠 버전 실험
                        - 차별점이 없는 경우 → 한가지라도 **핵심 포인트 만들기**
                        ''')   
                      
        elif cluster_option == '조회 편중형':
            st.markdown(f'#### 5️⃣ {cluster_option}')
            st.info('##### 조회 지표 최상이지만 전환율은 아쉬운 방송')
            with st.container():
                col1, col2, col3 = st.columns(3)
                col1.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">평균 지표</div>
                <div style="font-size:20px; font-weight:bold;">조회수 최상</div>
            </div>
            """, unsafe_allow_html=True)
                
                col2.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">방송 비율</div>
                <div style="font-size:20px; font-weight:bold;">6.02%</div>
            </div>
            """, unsafe_allow_html=True)

                col3.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">평균 매출</div>
                <div style="font-size:20px; font-weight:bold;">6,199,451원</div>
            </div>
            """, unsafe_allow_html=True)
                
                col4, col5, col6 = st.columns(3)
                col4.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">분당 유입</div>
                <div style="font-size:20px; font-weight:bold;">2394.85명</div>
            </div>
            """, unsafe_allow_html=True)

                col5.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">구매 고객</div>
                <div style="font-size:20px; font-weight:bold;">58.32명</div>
            </div>
            """, unsafe_allow_html=True)
                
                col6.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">실유입 대비 판매율</div>
                <div style="font-size:20px; font-weight:bold;">0.13%</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write('')
            
            st.markdown(f'''
                        ##### 🔍 원인 진단
                        - 유입은 높으나 **구매 전환율 낮음**
                        - 제품에 대한 **구매 설득력 부족**
                        - 정보 제공은 많으나 구매 유도 멘트 부족

                        ##### 전략 방향 ➡️ 전환율 개선
                        ''')        
            st.info('''                  
                        - 상품 정보 3번 반복 노출 (초반/중반/마무리)
                        - **구매 유도형 멘트** 삽입 (“지금 구매 시 OOO 증정” 등)
                        - 타사 상품과 비교 및 핵심 혜택 강조
                        - 후기, 별점, 실사용 사례 자막으로 실시간 노출
                        ''')
                       
        elif cluster_option == '전환 집중형':
            st.markdown(f'#### 5️⃣ {cluster_option}')
            st.info('##### 유입은 적지만 전환 효율이 뛰어난 알짜배기 방송')
            with st.container():

                col1, col2, col3 = st.columns(3)
                col1.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">평균 지표</div>
                <div style="font-size:20px; font-weight:bold;">전환 양호</div>
            </div>
            """, unsafe_allow_html=True)
                
                col2.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">방송 비율</div>
                <div style="font-size:20px; font-weight:bold;">19.65%</div>
            </div>
            """, unsafe_allow_html=True)

                col3.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">평균 매출</div>
                <div style="font-size:20px; font-weight:bold;">2,295,330원</div>
            </div>
            """, unsafe_allow_html=True)
                
                col4, col5, col6 = st.columns(3)
                col4.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">분당 유입</div>
                <div style="font-size:20px; font-weight:bold;">4.35명</div>
            </div>
            """, unsafe_allow_html=True)

                col5.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">구매 고객</div>
                <div style="font-size:20px; font-weight:bold;">14.20명</div>
            </div>
            """, unsafe_allow_html=True)
            
                col6.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">실유입 대비 판매율</div>
                <div style="font-size:20px; font-weight:bold;">6.72%</div>
            </div>
            """, unsafe_allow_html=True)

            st.write('')
                
            st.markdown(f'''
                        ##### 🔍 원인 진단
                        - **전환율은 높으나 유입이 적음**
                        - 방송 내용은 훌륭하나 홍보 부족

                        ##### 전략 방향 ➡️ 유입 확대 & 수익 극대화
                        ''')
            st.info('''                  
                        - 현재 구성 유지하며 **타겟 넓히기**
                        - 실시간 구매 알림, 채팅 하이라이트 등 상호작용 강화
                        - 동일 콘텐츠 반복 편성 (성과 유지 가능 시)
                        - 유입 극대화를 위한 **썸네일/제목 실험** 병행
                        ''')           
             
        elif cluster_option == '최다 매출형':
            st.markdown(f'#### 7️⃣ {cluster_option}')  
            st.success('##### 유입, 전환율, 매출 모두 높은 최우수 방송')
            with st.container():

                col1, col2, col3 = st.columns(3)
                col1.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">평균 지표</div>
                <div style="font-size:20px; font-weight:bold;">최우수</div>
            </div>
            """, unsafe_allow_html=True)
                
                col2.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">방송 비율</div>
                <div style="font-size:20px; font-weight:bold;">9.19%</div>
            </div>
            """, unsafe_allow_html=True)

                col3.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">평균 매출</div>
                <div style="font-size:20px; font-weight:bold;">6,451,718원</div>
            </div>
            """, unsafe_allow_html=True)
                
                col4, col5, col6 = st.columns(3)
                col4.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">분당 유입</div>
                <div style="font-size:20px; font-weight:bold;">5.82명</div>
            </div>
            """, unsafe_allow_html=True)

                col5.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">구매 고객</div>
                <div style="font-size:20px; font-weight:bold;">59.27명</div>
            </div>
            """, unsafe_allow_html=True)
                
                col6.markdown(f"""
            <div style=" padding:20px; border-radius:10px; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.05);">
                <div style="font-size:15px;">실유입 대비 판매율</div>
                <div style="font-size:20px; font-weight:bold;">42.46%</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write('')
                
            st.markdown(f'''
                        ##### 🔍 원인 진단
                        - **유입·전환·매출 모두 우수** → 최적 구조로 판단
                        - **현재 구성 유지**가 핵심

                        ##### 전략 방향 ➡️ 현상 유지 + 확장 고려
                        ''')
            
            st.success(''' 
                        - 성공 요소(멘트, 상품배치 등) 포맷화
                        - 다른 상품군에도 동일 구성 적용
                        - 시청 로그/리액션 분석 → 반복 가능한 요소 파악
                        - 클립 재활용 : **쇼츠/요약본 제작**해 리마케팅 활용
                        - 동일 MC/세트/콘셉트로 시리즈화 고려
                        ''')
  
    # 카테고리별 행동 지침 제공
    with wide2:

        st.markdown(f"""
        <div style='background-color:#DFF5E1; padding:10px 15px; border-radius:8px; margin-bottom:10px;'>
        <span style='font-weight:bold; font-size:22px;'>📁카테고리별 행동 지침</span>
        </div>
        """, unsafe_allow_html=True)
        
        category = st.selectbox(
                '카테고리를 선택하세요',
                ['패션의류', '화장품/미용', '패션잡화', '생활/건강', '식품', 
                '출산/육아', '여가/생활편의', '가구/인테리어', 
                '디지털/가전', '스포츠/레저', '도서'],
                index=0)
                
        tabs = st.tabs(['1️⃣ 가격 전략 추천', '2️⃣ 방송 시간 추천'])

        if category == '패션의류':
            with tabs[0]:
                with st.expander('💸 오픈방송 가격대 전략 가이드'):
                    st.markdown('''
                                - ✅ 동일 가격대 내에서도 방송유형(오픈/기획)에 따라 판매성과에 매우 큰 차이가 나타납니다.
                                - ✅ **오픈방송에서 이미 효과가 검증된 가격 구간을 강화**하고, **기획방송에서 비교적 부진하거나 미판매 중인 가격대를 틈새 공략**해보세요!
                                - ✅ 아래 안내된 오픈방송의 평균가와 상품 가격대 구성 전략을 참고하여, 카테고리별로 최적의 가격 전략을 수립해 보시기 바랍니다.
                                - 📍 **구성 설명**
                                    - 저가편중 : 상품 리스트 중 저가가 80% 이상
                                    - 고가편중 : 상품 리스트 중 고가가 80% 이상
                                    - 믹스형 : 다양한 가격대로 고루 구성''')
                    
                st.markdown(f'#### 👗 **{category}** 카테고리의 가격 전략 추천')
                st.markdown('''
                            - **판매량 기준 최적 구성**: 저가편중 & 믹스형
                            - **기준 가격**: 194,999원
                            - **추천 방송 평균가 : 30~50만원대**
                              - 50만원대는 기획방송에서 비교적 부진한 가격대입니다. 
                              - 가격 경쟁력이 있는 구간이니 오픈방송만의 상품 라인업을 강화해보세요.
                            
                            #### 💡TIP
                            
                            - 트렌드와 가격 모두에 민감하여 시즌별 프로모션·이벤트 구성이 유효합니다.\n
                            - **분당 평균 유입 수가 가장 적은 카테고리**로 확인됩니다. \n
                            - **2~5만원대의 저가 제품**으로 유입을 유도하고, **30만원 이상의 프리미엄 제품**을 **혼합 구성**하여 최적화 전략을 수립해보세요. \n
                            - 고가 의류는 재질, 브랜드, 코디 활용 등으로 **제품 및 브랜드력을 강조**해 가격 허들을 낮춰보세요.\n
                            - **수량 한정/단독 특가** 등으로 관심을 유도할 수도 있습니다.
                            '''
                         )
            with tabs[1]:
                with st.expander('🕒 오픈방송 시간대 전략 가이드'):
                    st.markdown('''
                                - ✅ 동일 상품이라도 시간대와 방송유형(오픈/기획)에 따라 판매성과는 달라집니다.
                                - ✅ 특히 기획방송이 강세인 구간을 피하고, 기획방송이 약하거나 부재한 시간대를 노리면 오픈방송만의 경쟁력을 확보할 수 있습니다.
                                - ✅ 상위30% 오픈 성과 시간대를 반복 편성하거나, 기획이 없는 ‘틈새 시간대’를 테스트해 정기 편성을 고려하세요!
                                ''')
                    
                st.markdown(f'#### 👗 **{category}** 카테고리의 방송 시간 추천')
                st.markdown('''
                            - **기획방송 성과 집중 시간대**
                                - 오전 9시 ~ 10시: 기획방송의 성과 피크 타임
                                - 이 시간대에는 경쟁이 치열하므로 오픈방송 편성은 피하는 것이 좋음
                            - **오픈방송 상위 시간대**
                                - 10시 이후 ~ 17시 이전: 오픈방송 성과가 비교적 좋은 구간
                                - 기획방송과의 간섭이 적고 전환율이 안정적임
                            - **최종 편성 가이드**
                                - 오전 피크 시간(9 ~ 10시)은 피하고, 오전 10시 이후 ~ 오후 5시 이전 편성이 유리
                                - 소분류별 특이 지점 존재하므로 타깃에 맞춰 심야 방송(23시 이후)도 고려하는 편이 좋음
                            
                            #### 💡TIP
                            
                            - **여성언더웨어/잠옷**은 **오전 9시대** 전환율이 54.3%로 매우 높았고,
                            **여성의류**는 **14시(3.6%)**, **남성의류**는 **23시(14.48%)** 에 성과 피크가 나타납니다.\n
                            - 소분류별 **전환율이 좋은 시간대를 선택적으로 활용**하는 전략이 효율적입니다. \n
                            - **전반적으로 낮은 전환율 구조**인 만큼, **성과가 입증된 시간대에 집중해 편성**하는 것이 좋습니다.'''
                         )
                
        elif category == '화장품/미용':
            
            with tabs[0]:
                with st.expander('#### 💸 오픈방송 가격대 전략 가이드'):
                    st.markdown('''
                                - ✅ 동일 가격대 내에서도 방송유형(오픈/기획)에 따라 판매성과에 매우 큰 차이가 나타납니다.
                                - ✅ **오픈방송에서 이미 효과가 검증된 가격 구간을 강화**하고, **기획방송에서 비교적 부진하거나 미판매 중인 가격대를 틈새 공략**해보세요!
                                - ✅ 아래 안내된 오픈방송의 평균가와 상품 가격대 구성 전략을 참고하여, 카테고리별로 최적의 가격 전략을 수립해 보시기 바랍니다.
                                - 📍 **구성 설명**
                                    - 저가편중 : 상품 리스트 중 저가가 80% 이상
                                    - 고가편중 : 상품 리스트 중 고가가 80% 이상
                                    - 믹스형 : 다양한 가격대로 고루 구성''')
                    
                st.markdown(f'#### 💄 **{category}** 카테고리의 가격 전략 추천')
                st.markdown('''
                            - **판매량 기준 최적 구성**: 믹스형
                            - **기준 가격**: 115,000원
                            - **추천 방송 평균가 : 20~40만원대**
                              - 40만원대는 파일럿 방송으로 시작해 성과에 따라 추후 편성 여부를 결정하는 것이 좋습니다.
                            
                            #### 💡TIP
                            - 가격 변화에 민감하며, 패키지 할인·구성 다양화로 매력도 제고가 효과적입니다.\n
                            - **가격대가 높은 스킨케어의 경우** 고가 디바이스와 젤 크림, 괄사 등의 부속품을 **패키지 구성**하거나 **저가 라인의 팩 세트를 추가**해 평균가를 낮춰보세요. 디바이스로만 구성했을 때보다 **판매량을 높일 수 있습니다.**\n
                            - 색조메이크업, 헤어스타일링, 뷰티소품 등의 **저가 소분류의 경우** **프리미엄 라인 제품을 함께 선별**해 방송 평균가를 높여보세요.\n
                            - 저가 제품은 입문용, 고가 제품은 전문용 또는 프리미엄 케어용 등으로 **가격대를 다양화해 구성**하면 고객의 **선택 폭이 넓어져** 장기적으로 더욱 효과적인 전략이 될 수 있습니다.
                            '''
                            )

            with tabs[1]:
                with st.expander('#### 🕒 오픈방송 시간대 전략 가이드'):
                    
                    st.markdown('''
                                - ✅ 동일 상품이라도 시간대와 방송유형(오픈/기획)에 따라 판매성과는 달라집니다.
                                - ✅ 특히 기획방송이 강세인 구간을 피하고, 기획방송이 약하거나 부재한 시간대를 노리면 오픈방송만의 경쟁력을 확보할 수 있습니다.
                                - ✅ 상위30% 오픈 성과 시간대를 반복 편성하거나, 기획이 없는 ‘틈새 시간대’를 테스트해 정기 편성을 고려하세요!
                                ''')
                    
                st.markdown(f'### 💄**{category}** 카테고리의 방송 시간 추천')
                
                st.markdown('''
                            - **기획방송 성과 집중 시간대**
                                - 기획방송과 오픈방송 간 성과 차이가 뚜렷하게 나타남
                                - 동일 소분류로의 경쟁은 비효율적일 수 있음
                            - **오픈방송 상위 시간대**
                                - 15시 이후로 편성할 경우 상대적으로 경쟁이 덜함
                                - 기획방송이 강세인 시간대를 피해 편성하는 전략 필요
                            - **최종 편성 가이드**
                                - 제품군을 차별화하거나 기획 피크타임을 피한 15시 이후 방송을 고려
                            
                            ### 💡TIP
                            
                            - 클렌징, 선케어, 네일케어는 **18~20시**에 오픈방송 전환율이 매우 높았습니다.\n
                            - 기획방송은 **13~15**시에 전환율이 집중됩니다.\n
                            - 성과 피크가 겹치지 않는 시간대 공략으로 **효과를 극대화**할 수 있습니다.
                            '''
                         )
                        
        elif category == '패션잡화':
            
            with tabs[0]:
                with st.expander('#### 💸 오픈방송 가격대 전략 가이드'):
                    st.markdown('''
                                - ✅ 동일 가격대 내에서도 방송유형(오픈/기획)에 따라 판매성과에 매우 큰 차이가 나타납니다.
                                - ✅ **오픈방송에서 이미 효과가 검증된 가격 구간을 강화**하고, **기획방송에서 비교적 부진하거나 미판매 중인 가격대를 틈새 공략**해보세요!
                                - ✅ 아래 안내된 오픈방송의 평균가와 상품 가격대 구성 전략을 참고하여, 카테고리별로 최적의 가격 전략을 수립해 보시기 바랍니다.
                                - 📍 **구성 설명**
                                    - 저가편중 : 상품 리스트 중 저가가 80% 이상
                                    - 고가편중 : 상품 리스트 중 고가가 80% 이상
                                    - 믹스형 : 다양한 가격대로 고루 구성''')
                    
                st.markdown(f'#### 🧢 **{category}** 카테고리의 가격 전략 추천')
                st.markdown('''
                            - **판매량 기준 최적 구성**: 저가 편중 & 믹스형
                            - **기준 가격**: 200,000원
                            - **추천 방송 평균가 : 최대 30만원대**
                              - 10만원~20만원대는 기획방송에서 비교적 부진한 가격대입니다. 
                              - 가격 경쟁력이 있는 구간이니 오픈방송만의 상품 라인업을 강화해보세요.
                            
                            #### 💡TIP
                            - 소액/충동 구매가 많아 저가 전략과 프로모션 병행이 유효합니다.\n
                            - 남성/여성 신발, 패션소품, 장갑, 모자, 양말 등 **저관여/저가 소분류**의 경우에는 **10~20만원대의 고가 제품**을 추가 셀렉해 평균가를 높여보세요. 
                            특히 **30만원대 방송**에서 타 구간 대비 큰 차이로 판매 효과가 높았던 것으로 확인됩니다. **프리미엄 라인을 고려**해봐도 좋겠습니다.\n
                            - **가격대별 기능 차이**와 **다양한 스타일링 예시**를 함께 제시하면, 새롭게 출시하는 고가 상품으로의 **구매 전환을 더욱 효과적**으로 이끌어낼 수 있습니다.\n
                            - 주얼리, 순금, 지갑 등 **고관여/고가 소분류**의 경우에는 **방송 평균가가 지나치게 높아지지 않도록 유의**하세요!\n
                            - 색조메이크업, 헤어스타일링, 뷰티소품 등의 **저가 소분류**의 경우 **프리미엄 라인 제품을 함께 선별**해 방송 평균가를 높여보세요.\n
                            - 저가 제품은 입문용, 고가 제품은 전문용 또는 프리미엄 케어용 등으로 **가격대를 다양화해 구성**하면 고객의 선택 폭이 넓어져 **장기적으로 더욱 효과적인 전략**이 될 수 있습니다.
                            '''
                            )

            with tabs[1]:
                with st.expander('#### 🕒 오픈방송 시간대 전략 가이드'):
                    st.markdown('''
                                - ✅ 동일 상품이라도 시간대와 방송유형(오픈/기획)에 따라 판매성과는 달라집니다.
                                - ✅ 특히 기획방송이 강세인 구간을 피하고, 기획방송이 약하거나 부재한 시간대를 노리면 오픈방송만의 경쟁력을 확보할 수 있습니다.
                                - ✅ 상위30% 오픈 성과 시간대를 반복 편성하거나, 기획이 없는 ‘틈새 시간대’를 테스트해 정기 편성을 고려하세요!
                                ''')
                    
                st.markdown(f'#### 🧢 **{category}** 카테고리의 방송 시간 추천')
                st.markdown('''
                            - **기획방송 성과 집중 시간대** 
                                - 오픈방송과 기획방송 간 성과 차이가 크지 않음  
                                - 전략적 시간대 선택 시 기획보다 우수한 성과 가능
                            - **오픈방송 상위 시간대**  
                                - 10시 ~ 15시 사이  
                                - 실유입 대비 판매율과 구매전환율이 안정적
                            - **최종 가이드**
                                - 10~15시 집중 편성 전략을 통해 안정적 성과 확보 가능
                            
                            #### 💡TIP
                            
                            - 양말, 헤어액세서리, 여성가방 등 **단가가 낮고 반복 구매 가능성이 높은 소분류**에서 전환율이 **특히 높게** 나타났습니다.\n
                            - 양말은 **구매고객 전환율**까지 높게 형성되어 있어 **소규모 예산으로도 성과를 극대화**할 수 있습니다.
                            '''
                         )

        elif category == '생활/건강':
            
            with tabs[0]:
                with st.expander('#### 💸 오픈방송 가격대 전략 가이드'):
                    st.markdown('''
                                - ✅ 동일 가격대 내에서도 방송유형(오픈/기획)에 따라 판매성과에 매우 큰 차이가 나타납니다.
                                - ✅ **오픈방송에서 이미 효과가 검증된 가격 구간을 강화**하고, **기획방송에서 비교적 부진하거나 미판매 중인 가격대를 틈새 공략**해보세요!
                                - ✅ 아래 안내된 오픈방송의 평균가와 상품 가격대 구성 전략을 참고하여, 카테고리별로 최적의 가격 전략을 수립해 보시기 바랍니다.
                                - 📍 **구성 설명**
                                    - 저가편중 : 상품 리스트 중 저가가 80% 이상
                                    - 고가편중 : 상품 리스트 중 고가가 80% 이상
                                    - 믹스형 : 다양한 가격대로 고루 구성''')
                    
                st.markdown(f'#### 🧹 **{category}** 카테고리의 가격 전략 추천')
                st.markdown('''
                            - **판매량 기준 최적 구성**: 믹스형
                            - **기준 가격**: 210,000원
                            - **추천 방송 평균가 : 60~80만원대**
                              - 70~80만원대는 파일럿 방송으로 시작해 성과에 따라 추후 편성 여부를 결정하는 것이 좋습니다.
                            
                            #### 💡TIP
                            - 실용성 제품 중심으로 세트 구성·가격 혜택 일부 활용이 가능합니다.\n
                            - 안마용품, 발건강용품, 악기 등 **고가 라인의 소분류**의 경우 비교적 저렴한 **저가 부속용품을 함께 구성**해 **방송 평균가가 지나치게 높아지지 않도록 유의**하세요!\n
                            - **고가 제품**은 **실제 리뷰나 사용 사례를 적극적으로 강조**하고, **저가 제품은 묶음 구성으로 판매**하면 고객의 구매 전환 및 만족도를 높일 수 있습니다.
                            '''
                            )

            with tabs[1]:                
                with st.expander('#### 🕒 오픈방송 시간대 전략 가이드'):
                    st.markdown('''
                                - ✅ 동일 상품이라도 시간대와 방송유형(오픈/기획)에 따라 판매성과는 달라집니다.
                                - ✅ 특히 기획방송이 강세인 구간을 피하고, 기획방송이 약하거나 부재한 시간대를 노리면 오픈방송만의 경쟁력을 확보할 수 있습니다.
                                - ✅ 상위30% 오픈 성과 시간대를 반복 편성하거나, 기획이 없는 ‘틈새 시간대’를 테스트해 정기 편성을 고려하세요!
                                ''')
                    
                st.markdown(f'#### 🧹 **{category}** 카테고리의 방송 시간 추천')
                st.markdown('''
                            - **기획방송 성과 집중 시간대**
                                - 기획방송이 전반적으로 높은 성과를 보임  
                                - 오전 11시 전후가 기획방송의 피크타임
                            - **오픈방송 상위 시간대**
                                - 오전 10시 ~ 12시  
                                - 밤 20시 ~ 21시  
                                - 해당 시간대에 높은 전환율과 실유입 대비 판매율 기록
                            - **최종 가이드**
                                - 기획방송 피크타임(오전 11시 전후)을 피해  
                                - 10~12시, 20~21시에 오픈방송 편성 집중
                            
                            #### 💡TIP
                            
                            - 특히 **반복 소비형 상품군**에서 오픈 방송의 **효율이 높게** 나타납니다.\n
                            - **재구매 가능성이 높은 품목을 중심으로 방송 기획을 강화**하는 것이 좋습니다.\n
                            - 이를 통해 구매전환뿐만 아니라 고객 충성도 확보까지 기대할 수 있습니다.
                            '''
                         )
                
                with st.expander('📍 소분류별 전략 더 보기'):
                    st.markdown('''
                                ##### 1️⃣ 수집품
                                - **기획 방송 강세 타임:** 10시,11시 특히 13시에 기획방송 성과가 월등히 높습니다.
                                - **오픈 방송 추천 타임:** 19시 → 실유입 대비 판매율과 구매전환율에서 일부 우위를 보입니다.
                                - **가이드:** 19시대를 우선 편성하되, 기획 강세 시간대는 피하는 전략으로 안정적인 성과 확보를 노리세요.
                                
                                ##### 2️⃣ 안마용품
                                - **기획 강세 시간인 10, 14, 22시는 피하세요.** 기획 성과 우위로 오픈 방송 경쟁력이 떨어질 수 있습니다.
                                - **오픈 방송 추천 타임:** **12 ~ 13시**, **15 ~ 20시** → 기획의 판매율/전환율이 확연히 낮고, 오픈방송과 구매 전환율이 비슷한 시간입니다.
                                - **가이드:** 기획 강세 시간대는 피하고 오픈방송의 최적화 방안을 모색해봐야해요. **20시**는 기획보다 구매전환율이 높지만, **22시의 기획 피크타임을 주의**하세요.
                                ''')
                    
        elif category == '식품':
            
            with tabs[0]:
                with st.expander('#### 💸 오픈방송 가격대 전략 가이드'):
                    st.markdown('''
                                - ✅ 동일 가격대 내에서도 방송유형(오픈/기획)에 따라 판매성과에 매우 큰 차이가 나타납니다.
                                - ✅ **오픈방송에서 이미 효과가 검증된 가격 구간을 강화**하고, **기획방송에서 비교적 부진하거나 미판매 중인 가격대를 틈새 공략**해보세요!
                                - ✅ 아래 안내된 오픈방송의 평균가와 상품 가격대 구성 전략을 참고하여, 카테고리별로 최적의 가격 전략을 수립해 보시기 바랍니다.
                                - 📍 **구성 설명**
                                    - 저가편중 : 상품 리스트 중 저가가 80% 이상
                                    - 고가편중 : 상품 리스트 중 고가가 80% 이상
                                    - 믹스형 : 다양한 가격대로 고루 구성''')
                    
                st.markdown(f'#### 🍔 **{category}** 카테고리의 가격 전략 추천')
                st.markdown('''
                            - **판매량 기준 최적 구성**: 믹스형
                            - **기준 가격**: 106,000원
                            - **추천 방송 평균가 : 10~40만원대**
                              - 30~40만원대는 파일럿 방송으로 시작해 성과에 따라 추후 편성 여부를 결정하는 것이 좋습니다.
                            
                            #### 💡TIP
                            - 가격에 민감하게 반응하므로 할인·타임딜 등 공격적인 가격 전략이 효과적입니다.\n
                            - 저가/고가편중보다 **믹스형에서 압도적으로 판매량이 높은 카테고**리로 확인됩니다. **다양한 가격대 상품을 함께 구성**하는 것을 추천합니다.\n
                            - 식용유/오일, 라면/면류, 과자/베이커리 등 **일상 식재료 관련 소분류**의 경우 **10만원대의 인기상품과 혼합 판매하는 전략이 효과적**입니다.\n
                            - 특히 **명절시즌에는 선물 세트 기획전**을 열어 **특별 패키지 상품**을 판매하며 고객의 구매전환을 유도해보세요.
                            '''
                            )

            with tabs[1]:
                with st.expander('#### 🕒 오픈방송 시간대 전략 가이드'):
                    st.markdown('''
                                - ✅ 동일 상품이라도 시간대와 방송유형(오픈/기획)에 따라 판매성과는 달라집니다.
                                - ✅ 특히 기획방송이 강세인 구간을 피하고, 기획방송이 약하거나 부재한 시간대를 노리면 오픈방송만의 경쟁력을 확보할 수 있습니다.
                                - ✅ 상위30% 오픈 성과 시간대를 반복 편성하거나, 기획이 없는 ‘틈새 시간대’를 테스트해 정기 편성을 고려하세요!
                                ''')
                    
                st.markdown(f'#### 🍔 **{category}** 카테고리의 방송 시간 추천')
                st.markdown('''
                            - **기획방송 성과 집중 시간대**  
                                - 축산물 13시 475.3%, 다이어트 11시 244.7% 등 일부 소분류에서 성과 급등  
                                - 해당 시간대를 피해 오픈방송 편성 필요
                            - **오픈방송 상위 시간대**  
                                - 오전 시간대 (전업주부 대상)  
                                - 점심시간  
                                - 퇴근 시간대
                            - **최종 가이드**  
                                - 기획방송이 강한 시간대를 피하고  
                                - 오전~점심~저녁 시간대에 전략적 편성 권장
                            
                            #### 💡TIP
                            
                            - 오픈 방송에서 **축산물 20시**(36.58%), **라면/면류 12시**(45.9%) 등 특정 소분류에서 탁월한 성과가 확인되어 
                            **인기 소분류 중심의 집중 편성**이 효과적입니다.
                            '''
                         )

                with st.expander('📍 소분류별 전략 더 보기'):
                    st.markdown('''
                                ##### 1️⃣ 건강식품
                                - 기획방송과 오픈방송의 시간대별 성과흐름은 비슷해요. 다만, 건강식품은 카테고리 특성상 기획력이 중요하게 작용해요.
                                - **오픈 방송 추천 타임:** **9~11시**는 오픈 방송 성과가 확연히 높은 시간대에요. 오픈 방송에서 편성해야한다면 이시간대 전후로 편성하는 것을 고려해보세요.
                                - **비추천 타임:** **14~18시, 20시**는 실유입 대비 판매율이 오르지 않는 경향이 있어요.
                                
                                ##### 2️⃣ 식용유/오일
                                - **테스트 가능한 시간대:** **16시~17시,19시 이후**는 오픈방송이 노려볼만한 틈새 시간대에요.
                                - 단, 오전 10시는 기획방송의 판매율 압도적이라 충돌 우려가 있어요. 약간 시간을 늦춰 **11시에 편성**하는 것을 추천합니다.
                                - **오픈 방송 추천 타임:** 기획방송이 없거나 상대적으로 약세를 보이는 **16~17시와 19시 이후 시간대**를 적극 공략하고, 성과가 주춤한 18시는 잠시 휴식 시간으로 활용하는 것을 권장합니다.
                                - **가이드:** **11시, 16~17시, 19시 이후에 집중 편성**하고, 성과가 확인되면 이 시간대를 정기 방송으로 활용하세요. 20시는 구매전환율이 높지만, 22시의 기획 피크타임을 주의하세요.
                                ''')
                    
        elif category == '출산/육아':
            
            with tabs[0]:
                with st.expander('#### 💸 오픈방송 가격대 전략 가이드'):
                    st.markdown('''
                                - ✅ 동일 가격대 내에서도 방송유형(오픈/기획)에 따라 판매성과에 매우 큰 차이가 나타납니다.
                                - ✅ **오픈방송에서 이미 효과가 검증된 가격 구간을 강화**하고, **기획방송에서 비교적 부진하거나 미판매 중인 가격대를 틈새 공략**해보세요!
                                - ✅ 아래 안내된 오픈방송의 평균가와 상품 가격대 구성 전략을 참고하여, 카테고리별로 최적의 가격 전략을 수립해 보시기 바랍니다.
                                - 📍 **구성 설명**
                                    - 저가편중 : 상품 리스트 중 저가가 80% 이상
                                    - 고가편중 : 상품 리스트 중 고가가 80% 이상
                                    - 믹스형 : 다양한 가격대로 고루 구성''')
                st.markdown(f'#### 🍼 **{category}** 카테고리의 가격 전략 추천')
                st.markdown('''
                            - **판매량 기준 최적 구성**: 믹스형 & 저가편중
                            - **기준 가격**: 156,999원
                            - **추천 방송 평균가 : 30~50만원대**
                                - 30만원대는 기획방송에서 비교적 부진한 가격대입니다. 가격 경쟁력이 있는 구간이니 오픈방송만의 상품 라인업을 강화해보세요.
                                - 50만원대는 파일럿 방송으로 시작해 성과에 따라 추후 편성 여부를 결정하는 것이 좋습니다.
                            
                            #### 💡TIP
                            - 구매 시 전문성, 후기, 안전성 등 신뢰 요소가 더 큰 영향을 미칩니다.
                            - 고가편중 대비 **믹스형에서 압도적으로 판매량이 높은 카테고리**로 확인됩니다. 방송 평균가가 지나치게 높아지지지 않도록 **다양한 가격대 상품을 함께 구성**하는 것에 유의하세요!\n
                            - 카시트, 유모차, 안전용품, 유아가구 등 **고관여/고가 제품은** 일상 용품이나 인형, 잡화, 간식 등 비교적 **저가 소분류 제품과 함께 제안**하여 유입 및 구매 접근성을 높여보세요.\n
                            - 고관여 제품들은 재질, 안정성, 기능에 대한 **신뢰할 수 있는 정보를 충분히 제공**하는 것이 중요합니다. 관련 자료와 상세 설명을 미리 준비해 전환율을 높여보세요.\n
                            - **출산 축하 세트, 육아 패키지** 등 테마 기획전도 추천합니다.
                            '''
                            )

            with tabs[1]:
                with st.expander('#### 🕒 오픈방송 시간대 전략 가이드'):
                    st.markdown('''
                                - ✅ 동일 상품이라도 시간대와 방송유형(오픈/기획)에 따라 판매성과는 달라집니다.
                                - ✅ 특히 기획방송이 강세인 구간을 피하고, 기획방송이 약하거나 부재한 시간대를 노리면 오픈방송만의 경쟁력을 확보할 수 있습니다.
                                - ✅ 상위30% 오픈 성과 시간대를 반복 편성하거나, 기획이 없는 ‘틈새 시간대’를 테스트해 정기 편성을 고려하세요!
                                ''')
                    
                st.markdown(f'#### 🍼 **{category}** 카테고리의 방송 시간 추천')
                st.markdown('''
                            - **기획방송 성과 집중 시간대**  
                                - 시간대별로 비교적 고른 성과 분포를 보임  
                                - 특정 피크타임 없음
                            - **오픈방송 상위 시간대**  
                                - 오전 9시, 밤 22시  
                            - **최종 가이드**  
                                -  14~17시는 전환율과 판매율 모두 낮으므로 회피  
                                - 오전과 밤 시간대를 전략적으로 활용
                                
                            #### 💡TIP
                            
                            -  **특히 이유식, 육아 소모품 등 반복 구매가 이루어지는 상품군은 전반적으로 높은 전환율을 보이고 있어, 
                            해당 품목을 중심으로 방송을 기획하는 것이 좋습니다.**
                            '''
                         )
                
                with st.expander('📍 소분류별 전략 더 보기'):
                    st.markdown('''
                                ##### 1️⃣ 완구/매트
                                - 오전 10~12시와 19시는 기획방송의 강세 구간이지만, 오픈 방송도 좋은 성과를 보인 시간대입니다.
                                - **오픈 방송 추천 타임:** 실유입 대비 판매율은 오픈 11~13시가 더 높았어요. 방송 전략 수립 시에는 유입 대비 실제 판매로 이어진 실유입 대비 판매율 지표를 꼭 참고하는 것을 권장합니다.
                                - **가이드:**  주고객층의 활동 시간이 뚜렷한 카테고리이므로, 새로운 시간대 탐색보다는 유망한 시간대에 기획력을 집중해 성과를 극대화하는 전략을 추천합니다.
                                
                                ##### 2️⃣ 외출용품
                                - 오전 10~11시는 기획과 오픈 모두 성과가 높지만, 특히 기획방송의 성과가 압도적입니다.
                                - **탐색 타임:** 15~20시를 이용해 진입 최적화 시간대를 탐색해보세요. 기획 방송은 15시 이전에 몰려있어요.
                                - **가이드:** 기획 미진입 + 오픈 성과 우수 구간인 **13시, 14시, 21시**를 적극 공략해보세요. **13 ~ 14시**는 실유입 대비 판매율 기준 오픈이 높거나 비슷했고, 
                                21시는 기획방송이  극히 적고 오픈에서 일정 수준의 전환 성과 존재한 구간이에요.**가이드:** **11시, 16 ~ 17시, 19시 이후**에 집중 편성하고, 성과가 확인되면 이 시간대를 정기 방송으로 활용하세요.
                                - 기획 강세 시간대는 피하고 오픈방송의 최적화 방안을 모색해봐야해요. 20시는 기획보다 구매전환율이 높지만, 22시의 기획 피크타임을 주의하세요.
                                ''')
                    
        elif category == '여가/생활편의':
            
            with tabs[0]:
                with st.expander('#### 💸 오픈방송 가격대 전략 가이드'):
                    st.markdown('''
                                - ✅ 동일 가격대 내에서도 방송유형(오픈/기획)에 따라 판매성과에 매우 큰 차이가 나타납니다.
                                - ✅ **오픈방송에서 이미 효과가 검증된 가격 구간을 강화**하고, **기획방송에서 비교적 부진하거나 미판매 중인 가격대를 틈새 공략**해보세요!
                                - ✅ 아래 안내된 오픈방송의 평균가와 상품 가격대 구성 전략을 참고하여, 카테고리별로 최적의 가격 전략을 수립해 보시기 바랍니다.
                                - 📍 **구성 설명**
                                    - 저가편중 : 상품 리스트 중 저가가 80% 이상
                                    - 고가편중 : 상품 리스트 중 고가가 80% 이상
                                    - 믹스형 : 다양한 가격대로 고루 구성''')
                    
                st.markdown(f'#### ✈️ **{category}** 카테고리의 가격 전략 추천')
                st.markdown('''
                            - **판매량 기준 최적 구성**: 믹스형
                            - **기준 가격**: 64,000원
                            - **추천 방송 평균가 : 10만원 미만**
                            
                            #### 💡TIP
                            - 일회성 소비 + 소액 위주로, 가격보단 체험·실시간성 강조가 중요합니다.\n
                            - 고가 숙박권/여행상품에 비교적 저가인 시설 입장권을 함께 구성한 패키지 방송을 기획해보세요.\n
                            - **베스트 리뷰 선정 이벤트**나 **라이브 방송 중 실시간 경품 이벤트** 등을 통해 리뷰 수를 늘리면, 상품에 대한 신뢰도도 함께 높일 수 있습니다.
                            '''
                            )

            with tabs[1]:
                with st.expander('#### 🕒 오픈방송 시간대 전략 가이드'):
                    st.markdown('''
                                - ✅ 동일 상품이라도 시간대와 방송유형(오픈/기획)에 따라 판매성과는 달라집니다.
                                - ✅ 특히 기획방송이 강세인 구간을 피하고, 기획방송이 약하거나 부재한 시간대를 노리면 오픈방송만의 경쟁력을 확보할 수 있습니다.
                                - ✅ 상위30% 오픈 성과 시간대를 반복 편성하거나, 기획이 없는 ‘틈새 시간대’를 테스트해 정기 편성을 고려하세요!
                                ''')
                    
                st.markdown(f'#### ✈️ **{category}** 카테고리의 방송 시간 추천')
                st.markdown('''
                            여가/생활편의 카테고리는 14시와 17시에 기획방송 편성 기록이 없어 이 시간을 틈새시간대로 공략할 수 있으며, 
                            오픈 방송에서 성과가 좋았던 9시와 15시대를 노려보는 것도 좋은 전략입니다.
                            
                            - **기획방송 성과 집중 시간대**  
                                - 기획방송 편성 기록이 적은 시간대 존재 (14시, 17시)
                            - **오픈방송 상위 시간대**  
                                - 오전 9시, 오후 15시  
                            - **최종 가이드**  
                                - 14시와 17시는 기획이 없어 오픈방송 편성에 유리한 틈새 시간대  
                                - 오전/오후 타깃별 전략 편성 추천
                                
                            #### 💡TIP
                            - 해외여행 소분류는 기획방송에서 이미 강세를 보이고 있으므로 기획방송 편성을 적극적으로 고려하는 것이 효과적입니다.
                            '''
                         )

        elif category == '가구/인테리어':
            
            with tabs[0]:
                with st.expander('#### 💸 오픈방송 가격대 전략 가이드'):
                    st.markdown('''
                                - ✅ 동일 가격대 내에서도 방송유형(오픈/기획)에 따라 판매성과에 매우 큰 차이가 나타납니다.
                                - ✅ **오픈방송에서 이미 효과가 검증된 가격 구간을 강화**하고, **기획방송에서 비교적 부진하거나 미판매 중인 가격대를 틈새 공략**해보세요!
                                - ✅ 아래 안내된 오픈방송의 평균가와 상품 가격대 구성 전략을 참고하여, 카테고리별로 최적의 가격 전략을 수립해 보시기 바랍니다.
                                - 📍 **구성 설명**
                                    - 저가편중 : 상품 리스트 중 저가가 80% 이상
                                    - 고가편중 : 상품 리스트 중 고가가 80% 이상
                                    - 믹스형 : 다양한 가격대로 고루 구성''')
                    
                st.markdown(f'#### 🛏️ **{category}** 카테고리의 가격 전략 추천')
                st.markdown('''
                            - **판매량 기준 최적 구성**: 믹스형 & 저가편중
                            - **기준 가격**: 308,000원
                            - **추천 방송 평균가 : 10~30만원, 60~80만원대**
                                - 10만원~20만원대는 기획방송에서 비교적 부진한 가격대입니다. 가격 경쟁력이 있는 구간이니 오픈방송만의 상품 라인업을 강화해보세요.
                                - 70~80만원대는 파일럿 방송으로 시작해 성과에 따라 추후 편성 여부를 결정하는 것이 좋습니다.
                                                        
                            #### 💡TIP
                            - 고관여·고단가 제품 특성상 가격 변화보다 브랜드, 품질, 설치 서비스 등 비가격 요소가 핵심입니다.\n
                            - 저가 소품류와 함께 20~30만원대의 **가성비 가구**를 중심으로 구성해보세요.\n
                            - 60만원 이상의 초고가 가구를 판매할 경우, **저가 소품류를 혼합**해 평균 가격대를 조정할 수 있습니다.\n
                            - 가구와 소품을 조화롭게 스타일링한 **시각 자료를 제공**해 구매 전환을 적극적으로 유도해보세요.\n
                            - 배송, 설치, A/S 등 부가 서비스의 혜택을 강조하는 것도 중요합니다.
                            '''
                            )

            with tabs[1]:
                with st.expander('#### 🕒 오픈방송 시간대 전략 가이드'):
                    st.markdown('''
                                - ✅ 동일 상품이라도 시간대와 방송유형(오픈/기획)에 따라 판매성과는 달라집니다.
                                - ✅ 특히 기획방송이 강세인 구간을 피하고, 기획방송이 약하거나 부재한 시간대를 노리면 오픈방송만의 경쟁력을 확보할 수 있습니다.
                                - ✅ 상위30% 오픈 성과 시간대를 반복 편성하거나, 기획이 없는 ‘틈새 시간대’를 테스트해 정기 편성을 고려하세요!
                                ''')
                    
                st.markdown(f'#### 🛏️ **{category}** 카테고리의 방송 시간 추천')
                st.markdown('''
                            
                            - **기획방송 성과 집중 시간대**  
                                - 오전 시간대 (경쟁이 치열함)
                            - **오픈방송 상위 시간대**  
                                - 오전 10시 ~ 14시: 실유입 대비 판매율이 가장 높음
                            - **최종 가이드**  
                                - 오전 시간대에는 차별화된 상품 설명과 실시간 소통으로 경쟁력 확보  
                                - **소가구·생활소품** 중심 기획을 강화하는 전략이 효과적
                            
                            #### 💡TIP
                            
                            - 가구/인테리어 상품은 단가, 설치, 공간 맥락 등 복합적 요소가 작용합니다.\n                      
                            - 라이브의 생생함을 활용해 **설치 과정**과 **활용 팁**을 구체적으로 보여주는 것이 중요합니다.\n
                            - 방송 전 **사전알림과 고객참여 이벤트를 적극 활용**해 높은 구매전환으로 연결될 수 있도록 준비하세요.
                            '''
                         )
                
        elif category == '디지털/가전':
            
            with tabs[0]:
                with st.expander('#### 💸 오픈방송 가격대 전략 가이드'):
                    st.markdown('''
                                - ✅ 동일 가격대 내에서도 방송유형(오픈/기획)에 따라 판매성과에 매우 큰 차이가 나타납니다.
                                - ✅ **오픈방송에서 이미 효과가 검증된 가격 구간을 강화**하고, **기획방송에서 비교적 부진하거나 미판매 중인 가격대를 틈새 공략**해보세요!
                                - ✅ 아래 안내된 오픈방송의 평균가와 상품 가격대 구성 전략을 참고하여, 카테고리별로 최적의 가격 전략을 수립해 보시기 바랍니다.
                                - 📍 **구성 설명**
                                    - 저가편중 : 상품 리스트 중 저가가 80% 이상
                                    - 고가편중 : 상품 리스트 중 고가가 80% 이상
                                    - 믹스형 : 다양한 가격대로 고루 구성''')
                    
                st.markdown(f'#### 💻 **{category}** 카테고리의 가격 전략 추천')
                st.markdown('''
                        - **판매량 기준 최적 구성**: 믹스형
                        - **기준 가격**: 448,999원
                        - **추천 방송 평균가 : 50~80만원대, 100만원 이상**
                            - 100만원 이상은 기획방송에서 비교적 부진한 가격대입니다. 가격 경쟁력이 있는 구간이니 오픈방송만의 상품 라인업을 강화해보세요.
                                                        
                        #### 💡TIP

                        - 가격 변화보단 성능, 보증 신뢰 기반의 비가격 전략이 중요합니다.\n
                        - PC, 카메라/캠코터용품 등 **고가 전자제품**을 중심으로, **저가 액세서리나 주변기기를 함께 구성**한 혼합 판매를 기획해보세요.\n
                        - 방송 평균가가 낮아질수록 판매량이 낮아지는 카테고리로 확인되니 **적정 평균가를 유지**하는 것에 특히 유의하세요!\n
                        - 가전제품은 배송, 설치, A/S까지 포함된 **서비스를 강조**하면 고객의 신뢰와 구매 전환율을 높일 수 있습니다.
                            '''
                            )

            with tabs[1]:
                with st.expander('#### 🕒 오픈방송 시간대 전략 가이드'):
                    st.markdown('''
                                - ✅ 동일 상품이라도 시간대와 방송유형(오픈/기획)에 따라 판매성과는 달라집니다.
                                - ✅ 특히 기획방송이 강세인 구간을 피하고, 기획방송이 약하거나 부재한 시간대를 노리면 오픈방송만의 경쟁력을 확보할 수 있습니다.
                                - ✅ 상위30% 오픈 성과 시간대를 반복 편성하거나, 기획이 없는 ‘틈새 시간대’를 테스트해 정기 편성을 고려하세요!
                                ''')
                    
                st.markdown(f'#### 💻 **{category}** 카테고리의 방송 시간 추천')
                st.markdown('''
                            - **기획방송 성과 집중 시간대**  
                                - 기획과 오픈 모두 비슷한 성과 흐름  
                            - **오픈방송 상위 시간대**  
                                - 오전 10시 ~ 11시: 구매전환율 최고
                            - **최종 가이드**  
                                - 오전 10~11시 집중 편성 추천  
                                - 기획 방송과 유사한 성과를 기대할 수 있는 14~15시대도 고려
                                
                            #### 💡TIP
                            - 특히 저장장치, 음향가전, 액세서리와 같이 **즉시 구매 가능한 실용적이고 단순한 저관여 상품**을 중심으로 편성해 **기획방송과 차별화된 전략**을 세우는 것이 효과적입니다.
                            '''
                         )         
                
        elif category == '스포츠/레저':
            
            with tabs[0]:
                with st.expander('#### 💸 오픈방송 가격대 전략 가이드'):
                    st.markdown('''
                                - ✅ 동일 가격대 내에서도 방송유형(오픈/기획)에 따라 판매성과에 매우 큰 차이가 나타납니다.
                                - ✅ **오픈방송에서 이미 효과가 검증된 가격 구간을 강화**하고, **기획방송에서 비교적 부진하거나 미판매 중인 가격대를 틈새 공략**해보세요!
                                - ✅ 아래 안내된 오픈방송의 평균가와 상품 가격대 구성 전략을 참고하여, 카테고리별로 최적의 가격 전략을 수립해 보시기 바랍니다.
                                - 📍 **구성 설명**
                                    - 저가편중 : 상품 리스트 중 저가가 80% 이상
                                    - 고가편중 : 상품 리스트 중 고가가 80% 이상
                                    - 믹스형 : 다양한 가격대로 고루 구성''')
                    
                st.markdown(f'#### 🏋️‍♂️ **{category}** 카테고리의 가격 전략 추천')
                st.markdown('''
                        - **판매량 기준 최적 구성**: 믹스형 & 저가편중
                        - **기준 가격**: 117,000원
                        - **추천 방송 평균가 : 30만원대, 60만원대**
                                                        
                        #### 💡TIP
                        - 시즌성 소비 중심으로, 가격 변화보단 한정성·가치 제안 전략이 효과적입니다.\n
                        - 11만원 이하의 비교적 **저가 상품을 판매***하는 셀러라면 **추천 평균가를 참고**해 상품 라인업을 확장해보시기를 추천합니다.\n
                        - **30만원대**가 가장 효과적인 판매량을 보인 구간으로 확인되므로 중가 상품을 우선으로 추가 셀렉해보세요.\n
                        - 자전거, 오토바이/스쿠터 등은 **보호용품 소분류와 패키지 방송으로 기획**하는 것이 판매량을 더욱 높일 수 있습니다.
                            '''
                            )

            with tabs[1]:
                with st.expander('#### 🕒 오픈방송 시간대 전략 가이드'):
                    st.markdown('''
                                - ✅ 동일 상품이라도 시간대와 방송유형(오픈/기획)에 따라 판매성과는 달라집니다.
                                - ✅ 특히 기획방송이 강세인 구간을 피하고, 기획방송이 약하거나 부재한 시간대를 노리면 오픈방송만의 경쟁력을 확보할 수 있습니다.
                                - ✅ 상위30% 오픈 성과 시간대를 반복 편성하거나, 기획이 없는 ‘틈새 시간대’를 테스트해 정기 편성을 고려하세요!
                                ''')
                    
                st.markdown(f'#### 🏋️‍♂️ **{category}** 카테고리의 방송 시간 추천')
                st.markdown('''
                          
                            - **기획방송 성과 집중 시간대**  
                                - 오전 9시 ~ 10시: 명확한 피크타임 형성
                            - **오픈방송 상위 시간대**  
                                - 밤 20시 ~ 22시: 오픈 방송에서 좋은 성과
                            - **최종 가이드**  
                                - 10시 이후 성과 감소 → 오전 or 저녁 시간대에 집중 편성
                            #### 💡TIP
                            
                            - 특히 액세서리류, 운동 보조용품, 홈트용품 등 **단순하고 구매 결정이 빠른 상품군**에서 **높은 전환율**이 나타납니다.\n
                            - 자전거나 스포츠/헬스 장비처럼 **고가이거나 체험성이 중요한 소분류**는 **기획방송**의 성과가 압도적입니다\n
                            - 상품 특성에 따라 기획 또는 오픈 전략을 구분해 접근하는 것이 효과적입니다.
                            '''
                         )
                
        elif category == '도서':
            
            with tabs[0]:
                with st.expander('#### 💸 오픈방송 가격대 전략 가이드'):
                    st.markdown('''
                                - ✅ 동일 가격대 내에서도 방송유형(오픈/기획)에 따라 판매성과에 매우 큰 차이가 나타납니다.
                                - ✅ **오픈방송에서 이미 효과가 검증된 가격 구간을 강화**하고, **기획방송에서 비교적 부진하거나 미판매 중인 가격대를 틈새 공략**해보세요!
                                - ✅ 아래 안내된 오픈방송의 평균가와 상품 가격대 구성 전략을 참고하여, 카테고리별로 최적의 가격 전략을 수립해 보시기 바랍니다.
                                - 📍 **구성 설명**
                                    - 저가편중 : 상품 리스트 중 저가가 80% 이상
                                    - 고가편중 : 상품 리스트 중 고가가 80% 이상
                                    - 믹스형 : 다양한 가격대로 고루 구성''')
                    
                st.markdown(f'#### 📚 **{category}** 카테고리의 가격 전략 추천')
                st.markdown('''
                        - **판매량 기준 최적 구성**: 믹스형
                        - **기준 가격**: 60,000원
                        - **추천 방송 평균가 : 최대 40만원**
                            - 30~40만원대는 파일럿 방송으로 시작해 성과에 따라 정식 판매 여부를 결정하는 것이 좋습니다.
                            
                        #### 💡TIP
                        - 콘텐츠 수요 중심이나, 가격 변화보단 추천·리뷰·주제 큐레이션 전략이 효과적입니다.
                        - **단권보다 다양한 상품을 패키지 구성**하여 **평균가가 높아졌을 때 판매량이 높습니다.**\n
                        - 수강권과 과목별 문제집을 함께 구성하거나, 유아 그림책과 그림펜 세트 등 다양한 패키지 방송을 기획해보세요.\n
                        - CD, 포스터, 낱말카드, 케이스 등 **부속상품의 퀄리티도 함께 강조**하면 상품의 매력을 더욱 높일 수 있습니다.
                            '''
                            )

            with tabs[1]:
                with st.expander('#### 🕒 오픈방송 시간대 전략 가이드'):
                    st.markdown('''
                                - ✅ 동일 상품이라도 시간대와 방송유형(오픈/기획)에 따라 판매성과는 달라집니다.
                                - ✅ 특히 기획방송이 강세인 구간을 피하고, 기획방송이 약하거나 부재한 시간대를 노리면 오픈방송만의 경쟁력을 확보할 수 있습니다.
                                - ✅ 상위30% 오픈 성과 시간대를 반복 편성하거나, 기획이 없는 ‘틈새 시간대’를 테스트해 정기 편성을 고려하세요!
                                ''')
                    
                st.markdown(f'#### 📚 **{category}** 카테고리의 방송 시간 추천')
                st.markdown('''
 
                            - **기획방송 성과 집중 시간대**  
                                - 오전 전체 성과 우수  
                                - 오전 10시: 실유입 대비 판매율 급락 → 11시에 반등  
                                - 신규 셀러라면 해당 구간 전략적 공략 권장
                            - **오픈방송 상위 시간대**  
                                - 성과 우수: 오전 시간대 전반  
                                - 편성 회피 권장: 17~19시, 21시, 23시  
                                - 22시에 성과 급등했지만 특수 상황일 수 있어 참고용으로만 활용
                            - **최종 가이드**  
                                - 오전 시간대 집중 편성 권장  
                                - 10~11시 성과 흐름을 고려한 정밀 타이밍 전략 필요
                                
                            #### 💡TIP
                            
                            - 오픈 방송에서 전환율 상위 3개 방송은 모두 **유아도서**입니다.\n
                            - **오전 10~11시는 전업주부**, **밤 22시는 직장맘**이 주요 고객층으로 추정되니 **타깃 고객의 라이프스타일에 맞춘 시간대 편성**을 추천합니다.
                            '''
                         )       
                                                                                   
    #######################################################
    # 3. 제목 분석 결과 & 제목 추천 시스템
    st.divider()
    
    st.markdown(f"""
        <div style='background-color:#DFF5E1; padding:10px 15px; border-radius:8px; margin-bottom:10px;'>
        <span style='font-weight:bold; font-size:22px;'>📝 방송 제목 분석</span>
        </div>
        """, unsafe_allow_html=True)

    word_type = st.radio('📌 품사를 선택하세요', ['명사', '형용사'], horizontal=True)
    tab1, tab2 = st.tabs(['🔠 빈출 단어 워드클라우드', '📊 단어 영향력 순위'])
    
    
    if word_type == '명사':
        group = 'noun'
        title_high = '👍 우수 방송 명사'
        title_low = '👎 저성과 방송 명사'
        
        image_high = './card/우수 방송 명사.png'
        image_low = './card/저성과 방송 명사.png'
        
        image_title = '명사'
    else:
        group = 'adj'
        title_high = '👍 우수 방송 형용사'
        title_low = '👎 저성과 방송 형용사'
        
        image_high = './card/우수 방송 형용사.png'
        image_low = './card/저성과 방송 형용사.png'
        
        image_title = '형용사'
        
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'**👍 우수 방송의 최빈 {image_title}**')
            st.image(image_high, use_container_width=True)
        with col2:
            st.markdown(f'**👎 저성과 방송의 최빈 {image_title}**')  
            st.image(image_low, use_container_width=True)
        
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(title_high)
            
            word1 = word[word['group'] == f'{group}_high_tfidf']
            word1 = word1.sort_values(by='diff', ascending=False).head(10)
            fig1 = px.bar(word1, x='word', y='diff', labels={'word': '키워드', 'diff': '기여도'})
            fig1.update_traces(marker_color="#4CAF50")
            fig1.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10), showlegend=True)
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            st.markdown(title_low)
            word2 = word[word['group'] == f'{group}_low_tfidf']
            word2 = word2.sort_values(by='diff', ascending=True).head(10)
            
            fig2 = px.bar(word2, x='word', y='diff', labels={'word': '키워드', 'diff': '기여도'})
            fig2.update_traces(marker_color="#4CAF50")
            fig2.update_layout(yaxis=dict(autorange='reversed'))
            fig2.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10), showlegend=True)
            
            st.plotly_chart(fig2, use_container_width=True)
    
    st.divider()
    #### 제목 추천 시스템
    st.markdown(f"""
        <div style='background-color:#DFF5E1; padding:10px 15px; border-radius:8px; margin-bottom:10px;'>
        <span style='font-weight:bold; font-size:22px;'>✍️ 제목 추천 시스템</span>
        </div>
        """, unsafe_allow_html=True)
    
    left_col, right_col = st.columns([1, 1.5])  # 입력은 작게, 출력은 넉넉히

    with left_col:
        st.markdown('#### 📥 입력 정보')
        st.caption('''해당 요소들이 반영된 제목을 추천해드립니다!''')
        # 1. 상품명 입력
        product_name = st.text_input('상품명', placeholder='예: 떡볶이')
        
        # 2. 프로모션 강조 입력
        promotion = st.radio(
            '📦 프로모션이 포함되었나요?',
            ['아니오', '예']
            )
        
        # 포함 단어 리스트
        include_words = word[(word['group'] == 'noun_high_tfidf') | (word['group'] == 'adj_high_tfidf')]
        
        promotion_types = []
          
        # 2-1 강조일 경우       
        if promotion == '예':
            promotion_types = st.text_input('프로모션 유형 (쉼표로 구분)', placeholder='예: 초특가, 마감임박')
            # 프로모션 있으면   
            include_words = include_words[(include_words['label'] != '시즌') & (include_words['label'] != '카테고리')]

            include_list = include_words['word'].tolist()        
        # 만약에 프로모션 없으면
        else:
            # 프로모션 관련 단어 제외
            include_words = include_words[(include_words['label'] != '시즌') & (include_words['label'] != '카테고리') & (include_words['label'] != '프로모션')]
            
            include_list = include_words['word'].tolist()
            
        # 제외 단어 리스트
        exclude_words = word[(word['group'] == 'noun_low_tfidf') | (word['group'] == 'adj_low_tfidf')]
        exclude_words = exclude_words[(exclude_words['label'] != '시즌') & (exclude_words['label'] != '카테고리')]
        
        exclude_list = exclude_words['word'].tolist()
        
        # 3. 방송 목적
        purpose = st.text_input('📺 방송 목적을 입력해주세요 (쉼표로 구분)', placeholder='예) 신상품 출시, 재고 소진')

        # 4. 타깃 연령대
        target_age = st.selectbox(
            '👥 타깃 연령대를 선택해주세요',
            ['전 연령', '10대', '20대', '30대', '40대', '50대 이상']
            )

        # 5. 상품 특징 키워드
        product_features = st.text_input(
            '🛍️ 상품의 주요 특징 키워드를 입력해주세요. (쉼표로 구분)',
            placeholder='예) 스트레치, 여름용, 국내산'
            )

        # 최종 인풋 받기
        # 제목 생성 함수
        client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
        def generate_title(prompt):
            response = client.chat.completions.create(
                model='gpt-4',
                messages=[
                    {'role': 'system', 'content': '너는 쇼핑 라이브 방송의 제목을 만들어주는 카피라이팅 전문가야. 아래의 제약조건 및 입력문을 토대로 최고의 라이브 방송 제목을 만들어줘.'},
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0.8,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        if st.button('🚀 제목 추천 받기'):
            
            prompt = f'''
            너는 쇼핑라이브 제목 카피 전문가야.
            다음 조건에 맞춰 **총 3개의 제목**을 제시해줘.
            각 제목은 아래 3가지 스타일 중 하나씩 해당해야 하며, 제목 옆에 간단한 설명을 붙여줘.

            # 스타일
            1. 정보 전달 및 신뢰 강조형
            2. 감성 강조형
            3. 혜택/프로모션 강조형

            # 조건
            - 제시된 상품명을 반영해줘.
                - {product_name}
            - 프로모션에 입력된 단어를 제목에 포함해줘.
                - {', '.join(promotion_types) if promotion == '예' else '없음'}
            - 매출에 좋은 영향을 주는 단어
                - 해당 단어는 반드시 포함할 필요없고, 맥락에 맞는 단어가 있다면 사용해줘.
                - {', '.join(include_list)}
            - 반드시 제외할 단어
                - {', '.join(exclude_list)}
            - 제목 길이: 20자 이내
            - 방송 목적에 맞게 제목을 추천해줘.
                - {', '.join(purpose)}
            - 타깃 연령대에 맞는 단어를 사용해줘.
            - 상품 특징 키워드를 직접 제목에 사용해도 되고 의미가 유사한 단어를 사용해도 돼.
                - {', '.join(product_features)}
            
            # 출력 형식 예시
            1. “앵콜 특집! 호떡의 맛”
            [정보 전달 및 신뢰 강조형] : 앵콜 재방송이라는 말로 신뢰성 강조 + 상품 언급

            2. “따뜻한 간식, 추억 한입”
            [감성 강조형] : 감성적 단어로 정서적 공감 유도

            3. “지금 1+1, 단 하루!”
            [혜택/프로모션 강조형] : 혜택을 전면 배치해 구매 유도

            반드시 위와 같은 형식으로, 3개 모두 출력해줘.
            '''
            
            result = generate_title(prompt).strip()
            st.session_state.recommended_titles = [s for s in result.split('\n') if s.strip()]
            
    with right_col:
        st.markdown('#### 📤 추천된 제목 리스트')

        if 'recommended_titles' in st.session_state:
        # 하나의 success 박스에 제목 3개를 마크다운으로 묶어서 출력
            titles_markdown = '#### ✅ 추천 제목\n\n'
            for idx, title in enumerate(st.session_state.recommended_titles, 1):
                titles_markdown += f"###### {title.strip()}\n"
            st.success(titles_markdown)
        else:
            st.success('왼쪽에서 정보를 입력하고 "제목 추천 받기" 버튼을 눌러주세요.')
    
    st.divider()