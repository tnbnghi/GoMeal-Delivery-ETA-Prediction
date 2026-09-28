import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
import time
import plotly.express as px
import plotly.graph_objects as go

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="GoMeal - Hệ thống điều phối thông minh", layout="wide", initial_sidebar_state="expanded")

# --- 2. CUSTOM CSS (Thiết kế Premium) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800&family=Be+Vietnam+Pro:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Be Vietnam Pro', sans-serif;
    }
    h1, h2, h3, .header-text h1 {
        font-family: 'Montserrat', sans-serif;
        font-weight: 900;
        letter-spacing: -0.04em;
    }

    /* Base Font Size and Line Height */
    .stMarkdown, .stText, .stButton, p, li {
        font-size: 16px;
        line-height: 1.6;
    }

    .block-container { padding-top: 2rem; padding-bottom: 0rem; }

    /* Sidebar Styling Refinement */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        font-size: 15px !important;
        font-weight: 500 !important;
        color: #334155 !important;
        margin-bottom: 0px !important;
    }
    .sidebar .stSlider, .sidebar .stSelectbox, .sidebar .stNumberInput {
        margin-bottom: 15px !important;
    }

    /* Header Banner Premium */
    .header-banner {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        padding: 40px;
        border-radius: 24px;
        color: white;
        display: flex;
        align-items: center;
        margin-bottom: 35px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12);
        border-left: 10px solid #FFD700;
    }
    .header-logo { width: 90px; margin-right: 30px; filter: drop-shadow(0 6px 12px rgba(0,0,0,0.2)); }
    .header-text h1 { margin: 0; font-size: 42px; color: white; letter-spacing: -0.7px; font-weight: 700; line-height: 1.1; }
    .header-text p { margin: 8px 0 0 0; font-size: 19px; opacity: 0.9; font-weight: 300; }

    /* Glassmorphism KPI Cards (High Contrast) */
    .kpi-card {
        background: white;
        padding: 24px;
        border-radius: 20px;
        text-align: center;
        border: 1px solid #edf2f7;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        height: 190px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .kpi-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 50px rgba(0,0,0,0.1);
        border: 1px solid #cbd5e0;
    }
    .kpi-card h2 { margin: 10px 0; color: #1b5e20; font-size: 40px; font-weight: 800; line-height: 1; }
    .kpi-card p { margin: 0; color: #64748b; font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }

    /* Tabs Styling Premium */
    .stTabs [role="tablist"] { gap: 12px; margin-bottom: 25px; }
    .stTabs [role="tab"] {
        background-color: #f8fafc;
        border-radius: 14px;
        padding: 14px 24px;
        font-weight: 600;
        color: #64748b;
        border: 1px solid #e2e8f0;
        transition: all 0.3s;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1b5e20 !important;
        color: white !important;
        border-color: #1b5e20 !important;
        box-shadow: 0 6px 15px rgba(27, 94, 32, 0.25);
    }

    /* Action Buttons Custom */
    .stButton > button {
        border-radius: 14px !important;
        border: none !important;
        background: #1b5e20 !important;
        color: white !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        transition: all 0.3s !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(27, 94, 32, 0.3) !important;
        background: #2e7d32 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. BANNER ĐẦU TRANG (LOGO + TIÊU ĐỀ) ---
st.markdown(f"""
    <div class="header-banner">
        <img src="https://cdn-icons-png.flaticon.com/512/1037/1037762.png" class="header-logo">
        <div class="header-text">
            <h1>Hệ thống GoMeal Chuyên nghiệp</h1>
            <p>Phân tích sâu & Điều phối thông minh dựa trên trí tuệ nhân tạo (XAI)</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


import os

# --- 4. LOAD MODEL ---
@st.cache_resource
def load_my_model():
    try:
        # Lấy đường dẫn tuyệt đối của file model cùng thư mục với script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "catboost_reduced_model.cbm")
        
        model = CatBoostRegressor()
        model.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Lỗi load model: {e}")
        return None


model = load_my_model()

# --- Placeholder for Live Status Bar ---
status_placeholder = st.empty()

# --- 4.5. KHỞI TẠO TRẠNG THÁI (Session State) ---
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = 'landing' # 'landing' hoặc 'dashboard'
if 'last_eta' not in st.session_state:
    st.session_state.last_eta = 32

# --- 5. SIDEBAR: THIẾT KẾ MỚI ---
with st.sidebar:
    st.markdown("### 🎛️ Bảng Điều Khiển")
    st.divider()

    # --- Nhóm 1 ---
    st.markdown("#### 📦 THÔNG TIN ĐƠN HÀNG")
    distance = st.number_input("Khoảng cách giao hàng (kilômét)", 0.1, 50.0, 8.5)
    multiple_del_total = st.selectbox("Tổng số đơn hàng đang giao", [1, 2, 3, 4], index=2)
    multiple_del = str(float(multiple_del_total - 1))
    st.write("") 

    # --- Nhóm 2 ---
    st.markdown("#### 🚴 THÔNG TIN TÀI XẾ")
    driver_age = st.slider("Tuổi của tài xế", 18, 60, 25)
    driver_rating = st.slider("Điểm đánh giá tài xế (1-5 sao)", 1.0, 5.0, 4.2)

    vehicle_cond_options = {
        "Cũ": "0",
        "Trung bình": "1",
        "Tốt": "2",
        "Rất tốt": "3"
    }
    vehicle_cond_vi = st.selectbox("Tình trạng phương tiện", list(vehicle_cond_options.keys()), index=2)
    vehicle_cond = vehicle_cond_options[vehicle_cond_vi]
    st.write("")

    # --- Nhóm 3 ---
    st.markdown("#### 📍 KHU VỰC")
    city_type_options = {
        "Đô thị": "Urban",
        "Đô thị lớn": "Metropolitian",
        "Bán đô thị": "Semi-Urban"
    }
    city_type_vi = st.selectbox("Khu vực giao hàng", list(city_type_options.keys()))
    city_type = city_type_options[city_type_vi]

    festival_options = {
        "Không có lễ hội": "No",
        "Đang trong lễ hội": "Yes"
    }
    festival_vi = st.radio("Sự kiện lễ hội", list(festival_options.keys()))
    festival = festival_options[festival_vi]
    st.write("")

    # --- Nhóm 4 ---
    st.markdown("#### 🌦️ MÔI TRƯỜNG")
    weather_options = {
        "Nắng": "Sunny",
        "Nhiều mây": "Cloudy",
        "Mưa bão": "Stormy", # Dữ liệu gốc lưu là Stormy thay vì Rainy
        "Sương mù": "Fog",
        "Bão cát": "Sandstorms",
        "Gió": "Windy"
    }
    weather_vi = st.selectbox("Thời tiết", list(weather_options.keys()))
    weather = weather_options[weather_vi]

    traffic_options = {
        "Thấp": "Low",
        "Trung bình": "Medium",
        "Cao": "High",
        "Tắc nghẽn": "Jam"
    }
    traffic_vi = st.selectbox("Mật độ giao thông", list(traffic_options.keys()))
    traffic = traffic_options[traffic_vi]

    # --- Nhóm 5: THỜI GIAN ---
    st.markdown("#### 🕒 THỜI GIAN")
    
    # Khởi tạo thời gian mặc định nếu chưa có
    if 'order_time_input' not in st.session_state:
        st.session_state.order_time_input = pd.to_datetime("now").time()
        
    order_time = st.time_input("Thời gian đặt hàng", key='order_time_input')
    
    # Tính toán biến tuần hoàn cho giờ
    hour_float = order_time.hour + order_time.minute / 60.0
    hour_sin = np.sin(2 * np.pi * hour_float / 24)
    hour_cos = np.cos(2 * np.pi * hour_float / 24)

    st.write("")
    
    # Nút bấm chính
    if st.session_state.app_mode == 'landing':
        predict_btn = st.button("🚀 KÍCH HOẠT HỆ THỐNG", use_container_width=True)
        if predict_btn:
            st.session_state.app_mode = 'dashboard'
            st.rerun()
    else:
        if st.button("🏠 VỀ TRANG CHỦ", use_container_width=True):
            st.session_state.app_mode = 'landing'
            st.rerun()

# --- 6. XỬ LÝ DỰ BÁO TỰ ĐỘNG (REACTIVE) ---
# Thực hiện dự báo khi đang ở chế độ Dashboard
if st.session_state.app_mode == 'dashboard':
    # Logic Tính Phụ Phí sớm để KPI hiển thị
    surcharge = 0
    surcharge_reason = ""
    if weather != "Sunny":
        surcharge += 5000
        surcharge_reason = f"Thời tiết {weather_vi}"
    if traffic in ["High", "Jam"]:
        surcharge += 3000
        surcharge_reason += " & " if surcharge_reason else ""
        surcharge_reason += "Giao thông cao"

    weather_traffic = f"{weather}_{traffic}"
    traffic_city = f"{traffic}_{city_type}"

    input_data = pd.DataFrame({
        'weather_traffic': [weather_traffic], 
        'distance_km': [float(distance)],
        'delivery_person_age': [float(driver_age)], 
        'delivery_person_ratings': [float(driver_rating)],
        'vehicle_condition': [str(vehicle_cond)], 
        'traffic_city': [traffic_city],
        'multiple_deliveries': [str(multiple_del)], 
        'festival': [str(festival)],
        'hour_sin': [hour_sin],
        'hour_cos': [hour_cos]
    })

    if model:
        # Sắp xếp đúng thứ tự cột
        input_data = input_data[model.feature_names_]
        # Dự đoán thực tế
        st.session_state.last_eta = int(model.predict(input_data)[0])
        st.session_state.last_input_data = input_data 
    else:
        st.error("❌ KHÔNG TÌM THẤY MÔ HÌNH TRÍ TUỆ NHÂN TẠO!")
        st.stop()
else:
    # Nếu đang ở Landing, vẫn tính surcharge để hiển thị KPI mặc định (nếu cần)
    surcharge = 0
    surcharge_reason = ""

# Hiển thị KPI với giá trị thực tế/mặc định
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    if surcharge > 0:
        st.markdown(f'<div class="kpi-card" style="border: 2px solid #ff9800;"><p>🚚 GIAO HÀNG DỰ KIẾN</p><h2>{st.session_state.last_eta} phút</h2><p style="color:#ff9800; font-size:11px; font-weight:700;">(+{surcharge:,}đ PHỤ PHÍ)<br>{surcharge_reason}</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="kpi-card"><p>🚚 GIAO HÀNG TRUNG BÌNH</p><h2>{st.session_state.last_eta} phút</h2></div>', unsafe_allow_html=True)
with kpi2:
    st.markdown('<div class="kpi-card"><p>📦 ĐƠN ĐANG XỬ LÝ</p><h2>128</h2></div>', unsafe_allow_html=True)
with kpi3:
    st.markdown('<div class="kpi-card"><p>🚴 TÀI XẾ HOẠT ĐỘNG</p><h2>54</h2></div>', unsafe_allow_html=True)
with kpi4:
    # --- CẢNH BÁO BẢO TRÌ ---
    if vehicle_cond in ["0", "1"] and driver_rating >= 4.5:
        st.markdown('<div class="kpi-card" style="border: 2px solid #ff9800;"><p>🔧 XE XUỐNG CẤP</p><h2 style="color:#ff9800; font-size:20px;">12 tài xế ưu tú</h2><p style="font-size:11px; font-weight:700;">CẦN PHÁT MÃ BẢO TRÌ</p></div>', unsafe_allow_html=True)
    elif vehicle_cond in ["0", "1"]: 
        st.markdown('<div class="kpi-card" style="border: 2px solid #ff9800;"><p>🔧 CẢNH BÁO BẢO TRÌ</p><h2 style="color:#ff9800; font-size:24px;">Đề xuất hỗ trợ</h2></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="kpi-card"><p>⚠️ KHU VỰC CHẬM TRỄ</p><h2 style="color:red;">3</h2></div>', unsafe_allow_html=True)

# --- 7. HIỂN THỊ KẾT QUẢ CHI TIẾT ---
if st.session_state.app_mode == 'dashboard' and 'last_input_data' in st.session_state:
    input_data = st.session_state.last_input_data
    eta_val = st.session_state.last_eta

    # Xác định kịch bản vận hành
    if eta_val <= 35:
        scenario, color, color_rgba, status = "🟢 ỔN ĐỊNH", "#2e7d32", "rgba(46, 125, 50, 0.1)", "Hệ thống ổn định - Tối ưu lợi nhuận"
    elif eta_val <= 45:
        scenario, color, color_rgba, status = "🟡 CẦN THEO DÕI", "#f9a825", "rgba(249, 168, 37, 0.1)", "Nguy cơ chậm giao - Cần giám sát"
    else:
        scenario, color, color_rgba, status = "🔴 CẢNH BÁO", "#d32f2f", "rgba(211, 47, 47, 0.1)", "Cảnh báo quá tải - Can thiệp khẩn cấp"

    total_price = 35000 + surcharge # Giả định giá gốc 35.000 đồng

    # --- CẬP NHẬT LIVE STATUS BAR (Chiến lược 2) ---
    with status_placeholder:
        st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.1); border-left: 5px solid {color}; padding: 15px 25px; border-radius: 12px; margin-bottom: 25px; display: flex; justify-content: space-between; align-items: center; border: 1px solid #e2e8f0; background-color: white;">
                <div style="display: flex; align-items: center;">
                    <span style="font-size: 24px; margin-right: 15px;">{scenario.split()[0]}</span>
                    <div>
                        <b style="color: {color}; font-size: 16px;">TRẠNG THÁI HỆ THỐNG: {scenario.split()[1]}</b><br>
                        <small style="color: #64748b;">Mức độ tin cậy: 94.2% | Độ trễ: 45ms</small>
                    </div>
                </div>
                <div style="text-align: right;">
                    <b style="color: #1e293b;">{time.strftime("%H:%M:%S")}</b><br>
                    <small style="color: #64748b;">Giờ Tiêu chuẩn Việt Nam</small>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # --- CHUẨN BỊ DỮ LIỆU GIẢI THÍCH (GIÁ TRỊ SHAP) CHO CÁC THẺ ---
    pool = Pool(input_data, cat_features=model.get_cat_feature_indices())
    shap_values = model.get_feature_importance(pool, type='ShapValues')[0]
    base_value = shap_values[-1]
    feature_contributions = shap_values[:-1]
    
    feature_names_vi = {
        'delivery_person_age': 'Tuổi tài xế',
        'delivery_person_ratings': 'Đánh giá tài xế',
        'vehicle_condition': 'Tình trạng phương tiện',
        'multiple_deliveries': 'Số lượng đơn ghép',
        'festival': 'Sự kiện lễ hội',
        'distance_km': 'Khoảng cách giao',
        'weather_traffic': 'Thời tiết và giao thông',
        'traffic_city': 'Mật độ giao thông tại khu vực',
        'hour_combined': 'Thời gian đặt hàng'
    }
    
    final_features = []
    final_shaps = []
    hour_shap_sum = 0
    for col, val in zip(model.feature_names_, feature_contributions):
        if col in ['hour_sin', 'hour_cos']:
            hour_shap_sum += val
        else:
            final_features.append(col)
            final_shaps.append(val)
    
    final_features.append('hour_combined')
    final_shaps.append(hour_shap_sum)
    
    # Các yếu tố "nhạy cảm" hoặc nội bộ không nên show cho khách hàng
    customer_hidden_factors = ['delivery_person_age', 'delivery_person_ratings', 'vehicle_condition']

    # Tìm các yếu tố gây chậm trễ đáng kể (> 2 phút) và phù hợp để giải thích cho khách hàng
    significant_delays = [
        feature_names_vi.get(f, f).lower() 
        for f, v in zip(final_features, final_shaps) 
        if v > 2 and f not in customer_hidden_factors
    ]
    
    if significant_delays:
        if len(significant_delays) > 1:
            delay_reason_vi = ", ".join(significant_delays[:-1]) + " và " + significant_delays[-1]
        else:
            delay_reason_vi = significant_delays[0]
    else:
        # Nếu không có cái nào > 2 thì lấy cái cao nhất như cũ
        delay_reason_vi = feature_names_vi.get(final_features[np.argmax(final_shaps)], "yếu tố bối cảnh").lower()
    
    main_speedup_factor = final_features[np.argmin(final_shaps)]
    speedup_reason_vi = feature_names_vi.get(main_speedup_factor, main_speedup_factor)

    # --- 8. TẦNG 2: KẾT QUẢ DỰ BÁO & TRẠNG THÁI HỆ THỐNG ---
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        # Sử dụng nội dung HTML đã được Việt hóa 100%
        result_html = f"""
    <div style="background: white; padding: 40px 30px; border-radius: 24px; border: 1px solid #e2e8f0; box-shadow: 0 20px 50px rgba(0,0,0,0.05); text-align: center; position: relative; overflow: hidden;">
        <div style="position: absolute; top: 0; left: 0; width: 100%; height: 10px; background: {color};"></div>
        
        <p style="font-size: 13px; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 20px;">
            Dự báo thời gian giao
        </p>

        <h1 style="font-size: 88px; color: #1e293b; margin: 0; font-family: 'Montserrat', sans-serif; font-weight: 900; line-height: 1;">
            {eta_val} <span style="font-size: 28px; color: #475569; font-weight: 600; margin-left: 5px;">phút</span>
        </h1>
        
        <div style="margin: 25px auto; padding: 10px 20px; background: {color_rgba}; color: {color}; border-radius: 30px; display: inline-block; font-weight: 700; font-size: 14px;">
            {scenario}
        </div>

        <p style="font-size: 16px; color: #475569; margin-top: 10px; line-height: 1.6;">
            {status}
        </p>
        
        <hr style="border: 0; border-top: 1px solid #f1f5f9; margin: 30px 0;">
        
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="text-align: left;">
                <small style="color: #94a3b8; display: block; text-transform: uppercase; font-size: 10px; font-weight: 700;">Giá cước</small>
                <b style="color: #1e293b; font-size: 18px;">{total_price:,} đồng</b>
            </div>
            <div style="text-align: right;">
                <small style="color: #94a3b8; display: block; text-transform: uppercase; font-size: 10px; font-weight: 700;">Khoảng cách</small>
                <b style="color: #1e293b; font-size: 18px;">{distance} kilômét</b>
            </div>
        </div>
    </div>
    """
        st.html(result_html)

    with res_col2:
        tab1, tab2, tab3, tab4 = st.tabs(["📱 KHÁCH HÀNG", "🛵 TÀI XẾ", "⚙️ ĐIỀU PHỐI", "📈 QUẢN TRỊ"])

        with tab1:
            st.markdown(f"#### 🛍️ Trải nghiệm Giao hàng Cao cấp")
            
            # --- Premium Order Cards ---
            c_info1, c_info2 = st.columns(2)
            with c_info1:
                st.markdown(f"""
                <div style="background: white; padding: 20px; border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 15px;">
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <span style="font-size: 24px; margin-right: 10px;">🛍️</span>
                        <b style="color: #1e293b;">Chi tiết đơn hàng</b>
                    </div>
                    <p style="margin: 0; font-size: 14px; color: #64748b;">Mã đơn: #GOMEAL-{int(time.time()) % 10000}</p>
                    <p style="margin: 5px 0 0 0; font-size: 16px; color: #1e293b; font-weight: 600;">{total_price:,} đồng</p>
                </div>
                """, unsafe_allow_html=True)
            with c_info2:
                st.markdown(f"""
                <div style="background: white; padding: 20px; border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 15px;">
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <span style="font-size: 24px; margin-right: 10px;">🛵</span>
                        <b style="color: #1e293b;">Tài xế của bạn</b>
                    </div>
                    <p style="margin: 0; font-size: 14px; color: #64748b;">ID: DX-{driver_age}{int(driver_rating*10)}</p>
                    <p style="margin: 5px 0 0 0; font-size: 16px; color: #1e293b; font-weight: 600;">⭐ {driver_rating} | {driver_age} tuổi</p>
                </div>
                """, unsafe_allow_html=True)

            # --- Status Timeline (Visual Step) ---
            st.markdown(f"""
            <div style="margin-top: 10px; margin-bottom: 25px; background: #f8fafc; padding: 20px; border-radius: 16px; border: 1px solid #f1f5f9;">
                <p style="font-size: 12px; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 15px;">Trạng thái đơn hàng</p>
                <div style="display: flex; justify-content: space-between;">
                    <div style="text-align: center; flex: 1;"><div style="height: 6px; background: #1b5e20; border-radius: 3px; margin-bottom: 8px;"></div><small style="color:#1b5e20; font-weight:700;">Xác nhận</small></div>
                    <div style="text-align: center; flex: 1; margin: 0 10px;"><div style="height: 6px; background: #1b5e20; border-radius: 3px; margin-bottom: 8px;"></div><small style="color:#1b5e20; font-weight:700;">Chuẩn bị</small></div>
                    <div style="text-align: center; flex: 1;"><div style="height: 6px; background: #e2e8f0; border-radius: 3px; margin-bottom: 8px;"></div><small style="color:#94a3b8;">Đang giao</small></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if "ỔN ĐỊNH" in scenario:
                st.info("✨ **Ưu tiên cao.** Đơn hàng của bạn đang được xử lý qua lộ trình tối ưu nhất.")
            else:
                st.warning(f"🕒 **Xử lý bối cảnh.** Yếu tố {delay_reason_vi.lower()} đang được hệ thống điều phối xử lý để giảm thiểu độ trễ (+{max(final_shaps):.1f} phút).")
            
            st.markdown("---")
            if "CẢNH BÁO" in scenario:
                st.write(f"Chúng tôi trân trọng thời gian của bạn. Do **{delay_reason_vi.lower()}**, tài xế ưu tú của chúng tôi đang nỗ lực hết sức. Bạn nhận được ưu đãi 15% cho đơn hàng này.")
                if st.button("🎁 Nhận mã giảm giá bồi thường", key="btn_voucher_v2"):
                    st.toast("Đã thêm mã giảm giá vào ví của bạn!")
            else:
                st.write("Bữa ăn ngon của bạn đang dần hoàn thiện. Hãy sẵn sàng thưởng thức nhé! 🍽️")
            
            if st.button("📍 Kích hoạt hệ thống định vị GPS của tài xế", key="btn_track_v2"):
                st.info("Đang kết nối mã hóa với vệ tinh định vị toàn cầu của tài xế...")

        with tab2:
            st.markdown(f"#### 🛵 Chỉ dẫn cho Tài xế")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.info(f"⏱️ **Mục tiêu:** {eta_val} phút")
                st.write(f"Khoảng cách: **{distance} kilômét**")
            with col_d2:
                if weather_traffic == "Sunny_Low" or weather_traffic == "Cloudy_Low":
                    st.success("✅ Điều kiện lái xe lý tưởng.")
                else:
                    st.warning("⚠️ Chú ý: Đường trơn hoặc tầm nhìn hạn chế.")
            
            st.markdown("---")
            st.write("💡 **Lời khuyên an toàn & hiệu quả:**")
            if float(driver_rating) < 4.5:
                st.write("- Đừng quên mỉm cười và chào hỏi khách hàng để cải thiện đánh giá nhé! 🌟")
            if int(vehicle_cond) == 0:
                st.error("- **⚠️ CẢNH BÁO:** Xe của bạn đang ở tình trạng kém. Hãy di chuyển chậm và kiểm tra xe sau ca làm việc.")
            elif float(multiple_del) >= 1:
                st.write("- Bạn đang có nhiều đơn ghép, hãy sắp xếp lộ trình tối ưu nhất qua bản đồ.")
                if st.button("🗺️ Xem lộ trình tối ưu", key="btn_route"):
                    st.success("Đã tối ưu hóa lộ trình qua 3 điểm giao!")
            else:
                st.write("- Hãy duy trì tốc độ ổn định và tuân thủ luật giao thông để đảm bảo an toàn.")
            
            # --- Chiến lược 4: Bảo dưỡng ---
            if int(vehicle_cond) <= 1:
                st.divider()
                st.markdown("##### 🎁 Quà tặng bảo dưỡng")
                if driver_rating >= 4.0:
                    st.write("Nhờ thành tích lái xe tốt, bạn nhận được mã giảm giá bảo trì!")
                    if st.button("🛠️ Nhận mã giảm giá thay nhớt miễn phí", key="btn_maint"):
                        st.toast("Mã giảm giá: GOMEAL-CARE-2024 đã được gửi!")


        with tab3:
            st.markdown("### ⚙️ Trung tâm Điều hành Vận hành")
            
            # Phân tích điểm nghẽn chuyên nghiệp
            bottlenecks = [f"{feature_names_vi.get(f, f)} ({v:+.1f} phút)" for f, v in zip(final_features, final_shaps) if v > 2]
            
            col_ops1, col_ops2 = st.columns([1.5, 1])
            with col_ops1:
                st.markdown(f"""
                <div style="background: #fffafa; border: 1px solid #ffebee; padding: 20px; border-radius: 12px; margin-bottom: 20px;">
                    <b style="color: #c62828;">🚨 PHÁT HIỆN ĐIỂM NGHẼN NGUY CẤP:</b><br>
                    {"<br>".join([f"• {b}" for b in bottlenecks]) if bottlenecks else "• Không phát hiện yếu tố gây trễ nguy cấp."}
                </div>
                """, unsafe_allow_html=True)
                
                resilience_score = 'Cao' if not bottlenecks else 'Trung bình' if len(bottlenecks) < 3 else 'Thấp'
                st.write(f"**Chỉ số Phục hồi Hiện tại:** {resilience_score}")

            with col_ops2:
                st.markdown("**🔍 Bối cảnh Môi trường**")
                st.caption(f"Thời tiết: {weather_vi} | Giao thông: {traffic_vi}")
                st.write(f"Khu vực: **{city_type_vi}**")
                
                if st.button("🔔 Gửi cảnh báo vận hành", key="btn_warn_ops_unified"):
                    st.toast("Đã gửi cảnh báo ưu tiên cho toàn bộ chuỗi cung ứng.")

            # Radar năng lực tài xế (Keep radar as it was but more blended)
            st.markdown("**🎯 Ma trận Năng lực Tài xế**")
            # ... Radar Logic continues ...

            # --- 2.5️⃣ BẢN ĐỒ ĐIỀU PHỐI & RADAR ---
            col_map, col_radar = st.columns(2)
            with col_map:
                st.markdown("**🗺️ Bản đồ mô phỏng lộ trình**")
                
                # Tạo tọa độ ngẫu nhiên giả lập dựa theo loại thành phố
                if city_type == "Metropolitian":
                    base_lat, base_lon = 10.7769, 106.7009
                elif city_type == "Urban":
                    base_lat, base_lon = 10.8231, 106.6297
                else:
                    base_lat, base_lon = 10.8833, 106.8167
                    
                map_data = pd.DataFrame(
                    np.array([
                        [base_lat + np.random.uniform(-0.02, 0.02), base_lon + np.random.uniform(-0.02, 0.02)],
                        [base_lat + np.random.uniform(-0.05, 0.05), base_lon + np.random.uniform(-0.05, 0.05)]
                    ]),
                    columns=['lat', 'lon']
                )
                st.map(map_data, zoom=11, size=20)

            with col_radar:
                st.markdown("**🎯 Đánh giá năng lực tài xế (Điều phối thông minh)**")
                is_hard_order = distance > 8 or weather in ["Stormy", "Fog", "Sandstorms", "Windy"] or traffic in ["High", "Jam"]
                
                st.markdown(f"**Độ khó đơn hàng:** {'🔴 CAO (Cần tài xế ưu tú)' if is_hard_order else '🟢 CƠ BẢN (Dễ san sẻ)'}")
                
                req_exp = 5 if is_hard_order else 3
                req_veh = 3 if is_hard_order else 2
                req_rat = 4.8 if is_hard_order else 4.0
                
                real_exp = min(5, (driver_age - 18) / 8 + 1)
                real_veh = int(vehicle_cond)
                real_rat = driver_rating
                
                radar_df = pd.DataFrame(dict(r=[req_exp, req_veh, req_rat, req_exp], theta=['Kinh nghiệm', 'Chất lượng xe', 'Đánh giá', 'Kinh nghiệm'], Nhóm=['Yêu cầu']*4))
                radar_df2 = pd.DataFrame(dict(r=[real_exp, real_veh, real_rat, real_exp], theta=['Kinh nghiệm', 'Chất lượng xe', 'Đánh giá', 'Kinh nghiệm'], Nhóm=['Thực tế']*4))
                radar_data = pd.concat([radar_df, radar_df2])
                
                fig_radar = px.line_polar(radar_data, r='r', theta='theta', color='Nhóm', line_close=True, color_discrete_sequence=['#ff9800', '#2e7d32'], template="plotly_white")
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=True, height=250, margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig_radar, use_container_width=True)
                
                if is_hard_order and (driver_rating < 4.8 or int(vehicle_cond) < 2):
                    st.error("⚠️ Tài xế không phù hợp. Nguy cơ vi phạm cam kết chất lượng!")
                    if st.button("🔄 Điều phối tài xế mới (Ưu tú)", key="btn_reassign"):
                        st.toast("Đang tìm tài xế có điểm đánh giá > 4.8...")
                elif is_hard_order:
                    st.success("✅ Tài xế ưu tú phù hợp.")
                
                # Unify with global operational warning
                st.caption("Gửi thông báo nhanh cho nhà hàng qua radar")

            st.write("")

            # --- 4️⃣ GỢI Ý HÀNH ĐỘNG (Operational Recommendation) ---
            st.markdown("**💡 Gợi ý hành động cho vận hành**")

            if "ỔN ĐỊNH" in scenario:
                st.success("""
                    **Hệ thống đánh giá đơn hàng đang vận hành trong điều kiện thuận lợi.**  
                    **Khuyến nghị:**  
                    - Không cần can thiệp thủ công.  
                    - Tiếp tục điều phối tự động theo lộ trình hiện tại.
                """)
            elif "CẦN THEO DÕI" in scenario:
                st.warning("""
                    **Hệ thống phát hiện nguy cơ giao hàng chậm nhẹ.**  
                    **Khuyến nghị:**  
                    - Theo dõi lộ trình di chuyển của tài xế trên bản đồ.  
                    - Chuẩn bị phương án hỗ trợ (mã giảm giá) nếu thời gian giao tiếp tục tăng quá 10%.
                """)
            else:
                st.error("""
                    **NGUY CƠ VI PHẠM CAM KẾT (Dự báo giao hàng trễ nghiêm trọng).**  
                    **Khuyến nghị:**  
                    - Ưu tiên điều phối tài xế có kinh nghiệm (tài xế ưu tú) hoặc xe máy công suất lớn.  
                    - Theo dõi sát tiến trình lấy hàng tại nhà hàng để rút ngắn thời gian chờ.  
                    - Chuẩn bị sẵn kịch bản hỗ trợ khách hàng (gửi tin nhắn xin lỗi chủ động).
                """)

        with tab4:
            st.markdown("### 📈 Bảng Điều khiển Quản trị Chiến lược")
            
            st.info(f"💡 **Khuyến nghị Điều hành:** Trong kịch bản này, hãy tập trung tối ưu hóa **{delay_reason_vi}** để phục hồi ước tính {max(final_shaps):.1f} phút hiệu quả giao hàng.")

            col_m1, col_m2 = st.columns([1, 1.2])
            with col_m1:
                st.markdown("**📊 Cam kết chất lượng dịch vụ**")
    
                # SLA cố định 45 phút - tiêu chuẩn nghiệp vụ giao đồ ăn
                sla_threshold = 45

                # Tính xác suất vi phạm: eta/45 so với ngưỡng an toàn (75%)
                # < 35 phút: 0%, 35-45 phút: tăng dần 0-50%, > 45 phút: 50-100%
                if eta_val <= 35:
                    violation_prob = 0
                elif eta_val <= sla_threshold:
                    violation_prob = int((eta_val - 35) / (sla_threshold - 35) * 50)
                else:
                    violation_prob = min(100, 50 + int((eta_val - sla_threshold) * 4))

                viol_color = '#1b5e20' if violation_prob == 0 else '#f9a825' if violation_prob < 50 else '#c62828'

                st.markdown(f"""
                <div style="background: white; padding: 20px; border-radius: 16px; border: 1px solid #e2e8f0; margin-bottom: 15px;">
                    <small style="color: #64748b; font-weight:700;">XÁC SUẤT VI PHẠM CAM KẾT</small>
                    <h2 style="color: {viol_color}; margin:0;">{violation_prob}%</h2>
                    <p style="font-size: 13px; color: #475569; margin-top:5px;">Hạn mức cam kết: {sla_threshold} phút | Dự báo: {eta_val} phút</p>
                </div>
                """, unsafe_allow_html=True)
    
                # --- 2️⃣ PHÂN TÍCH NGUYÊN NHÂN & TÁC ĐỘNG ---
                st.markdown("**🔍 Tác động & Nguyên nhân**")
                impacts = []
                if weather != "Sunny": impacts.append(f"Thời tiết: {weather_vi}")
                if traffic in ["High", "Jam"]: impacts.append(f"Giao thông: {traffic_vi}")
                if distance > 7: impacts.append(f"Khoảng cách: Xa ({distance} kilômét)")
    
                if impacts:
                    st.write("Yếu tố làm tăng thời gian giao hàng: **" + ", ".join(impacts) + "**")
                else:
                    st.write("Yếu tố: Không phát hiện bất lợi ngoại cảnh.")
    
                if eta_val > 45:
                    st.error("📉 **TÁC ĐỘNG:** Nguy cơ đánh giá 1 sao cao. Khả năng phát sinh đền bù mã giảm giá.")
                elif eta_val > 35:
                    st.warning("⚠️ **TÁC ĐỘNG:** Khách hàng sốt ruột. Đe dọa biên lợi nhuận đơn hàng.")
                else:
                    st.success("📈 **TÁC ĐỘNG:** Đảm bảo mức độ hài lòng tối đa, tăng khả năng khách hàng quay lại.")

            with col_m2:
                # --- TỐI ƯU HIỆU QUẢ KINH TẾ ---
                st.markdown("**💡 Tối ưu hiệu quả kinh tế trên mỗi đơn hàng**")
                
                if float(multiple_del) >= 1:
                    st.warning(f"Tính năng kích hoạt mặc định do đang ghép {int(float(multiple_del)) + 1} đơn.")
                    enable_batching = True
                else:
                    enable_batching = st.toggle("Cho phép ghép đơn?", value=(traffic in ["Low", "Medium"]))
                
                if enable_batching:
                    base_rev = 20000 + (distance * 5000)
                    cost_single = base_rev * 0.8
                    profit_single = base_rev - cost_single
                    num_batch = int(float(multiple_del)) + 1
                    
                    if num_batch >= 2:
                        rev_batch = base_rev * num_batch * 0.9
                        cost_batch = base_rev * (num_batch * 0.5)
                        profit_batch = rev_batch - cost_batch
                        
                        st.markdown(f"""
                            <div style="padding:10px; border:1px solid #ddd; border-radius:10px; background:#fafafa; margin-bottom:10px;">
                                <h5 style="margin:0;">Giao dịch đơn lẻ mặc định</h5>
                                <p style="margin:0; font-size:13px; color:#555;">Dự báo giao: <b>{int(eta_val*0.8)} phút</b> | Lợi nhuận: <b style="color:#2e7d32;">{int(profit_single):,} đồng</b></p>
                            </div>
                            <div style="padding:10px; border:2px solid #2e7d32; border-radius:10px; background:#e8f5e9;">
                                <h5 style="margin:0; color:#2e7d32;">Chuyến ghép {num_batch} đơn hiện tại</h5>
                                <p style="margin:0; font-size:13px; color:#555;">Dự báo giao: <b>{eta_val} phút</b> | Doanh thu: {int(rev_batch):,} đồng</p>
                                <h4 style="margin:0; color:#2e7d32;">Lợi nhuận: {int(profit_batch):,} đồng <span style="font-size:12px; color:white; background:#2e7d32; padding:2px 5px; border-radius:3px;">+{(profit_batch/profit_single - 1)*100:.0f}%</span></h4>
                            </div>
                        """, unsafe_allow_html=True)
                        if traffic in ["High", "Jam"]:
                            st.error("⚠️ CẢNH BÁO: Kẹt xe tiềm ẩn rủi ro bùng nổ thời gian giao hàng. Tạm dừng ghép đơn ngay!")
                        else:
                            st.success(f"✅ HỆ THỐNG ĐANG TỐI ƯU HIỆU QUẢ KINH TẾ RẤT TỐT!")
                    else: 
                         num_batch = 3
                         rev_batch = base_rev * num_batch * 0.9
                         cost_batch = base_rev * (num_batch * 0.5)
                         profit_batch = rev_batch - cost_batch
                         st.markdown(f"""
                            <div style="padding:10px; border:1px solid #ddd; border-radius:10px; background:#fafafa; margin-bottom:10px;">
                                <h5 style="margin:0;'>Giao dịch đơn lẻ (hiện tại)</h5>
                                <p style="margin:0; font-size:13px; color:#555;">Dự báo giao: <b>{eta_val} phút</b> | Lợi nhuận: <b style="color:#2e7d32;">{int(profit_single):,} đồng</b></p>
                            </div>
                            <div style="padding:10px; border:2px dashed #ff9800; border-radius:10px; background:#fff8e1;">
                                <h5 style="margin:0; color:#ff9800;">Dự phóng tương lai: Nếu bật ghép 3 đơn</h5>
                                <p style="margin:0; font-size:13px; color:#555;">Dự báo giao: <b>{eta_val + 8} phút</b> (+8 phút rủi ro) | Doanh thu: {int(rev_batch):,} đồng</p>
                                <h4 style="margin:0; color:#2e7d32;">Lợi nhuận: {int(profit_batch):,} đồng <span style="font-size:12px; color:white; background:#2e7d32; padding:2px 5px; border-radius:3px;">+{(profit_batch/profit_single - 1)*100:.0f}%</span></h4>
                            </div>
                         """, unsafe_allow_html=True)
                         st.info("💡 Nên bật ghép đơn ở khu vực này để tối đa hóa biên lợi nhuận doanh nghiệp!")
                         if st.button("🔗 Kích hoạt ghép đơn ngay", key="btn_batch_now"):
                             st.success("Đã bật chế độ ghép đơn cho khu vực này!")
                else:
                     st.info("Chế độ ghép đơn đang tắt. Bật lại ở các khung giờ mật độ giao thông thấp.")
                
                st.divider()
                if surcharge > 0:
                    st.write(f"**Phụ phí hiện tại:** {surcharge:,} đồng")
                    if st.button("💹 Áp dụng phí linh hoạt toàn vùng", key="btn_apply_fee"):
                        st.toast("Đã áp dụng phụ phí cho toàn bộ khu vực bối cảnh tương tự.")

# --- 9. TẦNG 3 & 4: CHARTS & TABLES (PHÂN TÍCH CHUYÊN SÂU) ---
    st.divider()
    exp_analysis = st.expander("📊 Nhấn vào đây để xem phân tích dữ liệu chuyên sâu và nhật ký mô phỏng", expanded=False)
    exp_analysis.markdown(
        '<div style="font-size:20px; font-weight:bold; color:#1b5e20; margin-bottom:10px;">📉 HỆ THỐNG PHÂN TÍCH DỮ LIỆU ĐA CHIỀU</div>',
        unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SỬ DỤNG SHAP VALUES ĐỂ VẼ BIỂU ĐỒ PHÂN TÍCH
    chart_col1, chart_col2 = exp_analysis.columns(2)

    with chart_col1:
        exp_analysis.markdown("#### 📊 Mức độ ảnh hưởng của từng yếu tố (SHAP)")
        shap_df = pd.DataFrame({
            'Yếu tố': [feature_names_vi.get(f, f) for f in final_features],
            'Đóng góp (phút)': final_shaps
        }).sort_values('Đóng góp (phút)', ascending=True)

        colors = ['#d32f2f' if v > 0 else '#2e7d32' for v in shap_df['Đóng góp (phút)']]
        fig_waterfall = go.Figure(go.Bar(
            x=shap_df['Đóng góp (phút)'],
            y=shap_df['Yếu tố'],
            orientation='h',
            marker_color=colors
        ))
        fig_waterfall.update_layout(
            height=350,
            margin=dict(t=20, b=20, l=20, r=20),
            xaxis_title="Tác động (phút)",
            template="simple_white"
        )
        exp_analysis.plotly_chart(fig_waterfall, use_container_width=True)

    with chart_col2:
        exp_analysis.markdown("#### 📈 Xu hướng thời gian giao hàng theo giờ")
        hours = list(range(0, 24))
        # Công thức mô phỏng: ETA cao vào tầm 11-12h và 17-19h
        eta_trend = [eta_val + int(10 * np.sin((h - 8) * 0.5)) for h in hours]

        fig_line = px.line(
            x=hours, y=eta_trend,
            labels={'x': 'Giờ trong ngày', 'y': 'Thời gian giao hàng dự kiến (phút)'},
            template="simple_white"
        )
        fig_line.update_traces(line_color='#2e7d32', line_width=3, mode='lines+markers')
        fig_line.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
        exp_analysis.plotly_chart(fig_line, use_container_width=True)
        exp_analysis.caption("ℹ️ **Thông tin**: Biểu đồ thể hiện xu hướng biến động dự kiến theo giờ trong ngày.")

    # --- NHẬT KÝ MÔ PHỎNG ---
    exp_analysis.markdown("#### 📋 Nhật ký mô phỏng đơn hàng")

    # Khởi tạo history trong session_state nếu chưa có
    if 'simulation_history' not in st.session_state:
        st.session_state.simulation_history = []

    # Thêm kết quả hiện tại vào lịch sử (LƯU TOÀN BỘ CÁC BIẾN CHO FILE CSV)
    new_entry = {
        'Thời điểm': time.strftime("%H:%M:%S"),
        'Khoảng cách': f"{distance} kilômét",
        'Tuổi tài xế': driver_age,
        'Đánh giá tài xế': driver_rating,
        'Tình trạng xe': vehicle_cond_vi,
        'Số đơn ghép': multiple_del_total,
        'Lễ hội': festival_vi,
        'Khu vực': city_type_vi,
        'Thời điểm đặt': order_time.strftime("%H:%M"),
        'Thời tiết': weather_vi,
        'Giao thông': traffic_vi,
        'Thời gian dự báo': f"{eta_val} phút",
        'Kịch bản vận hành': scenario
    }
    # Chỉ thêm nếu chưa có trong lịch sử (tránh trùng khi streamlit rerun)
    if not st.session_state.simulation_history or st.session_state.simulation_history[0]['Thời điểm'] != new_entry[
        'Thời điểm']:
        st.session_state.simulation_history.insert(0, new_entry)

    # Hiển thị bảng nhật ký (Giới hạn 5 đơn gần nhất)
    df_history = pd.DataFrame(st.session_state.simulation_history)
    df_display = df_history[['Thời điểm', 'Khoảng cách', 'Thời tiết', 'Giao thông', 'Thời gian dự báo', 'Kịch bản vận hành']].head(5)
    exp_analysis.table(df_display)
    
    # --- NÚT TẢI NHẬT KÝ ---
    csv_data = df_history.to_csv(index=False).encode('utf-8')
    exp_analysis.download_button(
        label="📥 Tải nhật ký mô phỏng (tệp .csv)",
        data=csv_data,
        file_name='nhat_ky_mo_phong_gomeal.csv',
        mime='text/csv',
        help="Xuất danh sách các đơn hàng mô phỏng gần đây"
    )

else:
    # --- LANDING PAGE (Giao diện chào mừng khi chưa nhấn Predict) ---
    st.markdown("---")
    col_l1, col_l2, col_l3 = st.columns([1, 4, 1])
    with col_l2:
        st.markdown("""
        <div style="text-align: center; padding: 80px 40px; background: linear-gradient(135deg, #ffffff 0%, #f1f8e9 100%); border-radius: 30px; margin-top: 20px; border: 1px solid #e0e0e0; box-shadow: 0 15px 40px rgba(0,0,0,0.03);">
            <div style="display: inline-block; padding: 20px; background: #1b5e2015; border-radius: 30px; margin-bottom: 35px; animation: pulse 2s infinite;">
                <img src="https://cdn-icons-png.flaticon.com/512/1037/1037762.png" width="130">
            </div>
            <style>
                @keyframes pulse {
                    0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(27, 94, 32, 0.2); }
                    70% { transform: scale(1.05); box-shadow: 0 0 0 15px rgba(27, 94, 32, 0); }
                    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(27, 94, 32, 0); }
                }
            </style>
            <h1 style="font-size: 64px; color: #1e293b; margin-bottom: 25px; font-family: 'Montserrat', sans-serif; font-weight: 900; letter-spacing: -2px; line-height: 1.1;">
                Vận hành Thông minh <br><span style="color: #1b5e20;">Nâng tầm Vận tải</span>
            </h1>
            <p style="font-size: 22px; color: #475569; max-width: 800px; margin: 0 auto 50px auto; line-height: 1.7; font-family: 'Inter', sans-serif;">
                Kế thừa sức mạnh của <b>Trí tuệ Nhân tạo Có thể giải thích (XAI)</b>, GoMeal giúp bạn làm chủ thời gian giao hàng, tối ưu lợi nhuận và kiến tạo trải nghiệm khách hàng vượt trội.
            </p>
            <div style="display: flex; justify-content: center; gap: 30px; margin-top: 40px; flex-wrap: wrap;">
                <div style="background: white; padding: 40px 30px; border-radius: 24px; width: 280px; box-shadow: 0 10px 25px rgba(0,0,0,0.04); border: 1px solid #f0f4f8; transition: all 0.3s;">
                    <div style="font-size: 40px; margin-bottom: 15px;">📊</div>
                    <h3 style="color: #1b5e20; margin-bottom: 12px; font-family: 'Outfit', sans-serif; font-weight: 700;">Dự báo Chính xác</h3>
                    <p style="font-size: 15px; color: #64748b; line-height: 1.5;">Phân tích đa biến thời gian thực với độ chính xác trên 94%.</p>
                </div>
                <div style="background: white; padding: 40px 30px; border-radius: 24px; width: 280px; box-shadow: 0 10px 25px rgba(0,0,0,0.04); border: 1px solid #f0f4f8; transition: all 0.3s;">
                    <div style="font-size: 40px; margin-bottom: 15px;">🧠</div>
                    <h3 style="color: #1b5e20; margin-bottom: 12px; font-family: 'Outfit', sans-serif; font-weight: 700;">Hành động Gợi ý</h3>
                    <p style="font-size: 15px; color: #64748b; line-height: 1.5;">Đưa ra khuyến nghị cụ thể cho từng kịch bản vận hành cao điểm.</p>
                </div>
                <div style="background: white; padding: 40px 30px; border-radius: 24px; width: 280px; box-shadow: 0 10px 25px rgba(0,0,0,0.04); border: 1px solid #f0f4f8; transition: all 0.3s;">
                    <div style="font-size: 40px; margin-bottom: 15px;">💰</div>
                    <h3 style="color: #1b5e20; margin-bottom: 12px; font-family: 'Outfit', sans-serif; font-weight: 700;">Tối ưu Lợi nhuận</h3>
                    <p style="font-size: 15px; color: #64748b; line-height: 1.5;">Cân bằng hoàn hảo giữa chi phí vận hành và sự hài lòng của khách hàng.</p>
                </div>
            </div>
            <div style="margin-top: 70px; padding: 15px 30px; background: #f8fafc; border-radius: 50px; display: inline-block; color: #64748b; font-size: 15px; font-family: 'Inter', sans-serif; border: 1px dashed #cbd5e1;">
                👈 Bắt đầu bằng cách thiết lập thông số tại <b>Thanh bên</b> và nhấn <b>Kích hoạt hệ thống</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)
