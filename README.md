# 🛵 GoMeal - Ứng Dụng Phân Tích Dữ Liệu Trong Dự Đoán Thời Gian Giao Hàng (ETA)

## 📌 Giới thiệu dự án

Trong ngành giao đồ ăn trực tuyến, ước tính chính xác **thời gian giao hàng (ETA - Estimated Time of Arrival)** giúp quản trị kỳ vọng khách hàng, tối ưu vận hành và nâng cao lợi thế cạnh tranh.

Dự án xây dựng mô hình học máy cho nền tảng **GoMeal**, dự đoán chặng giao hàng cuối (từ lúc tài xế nhận đơn đến khi giao cho khách hàng) dựa trên các yếu tố như thời tiết, giao thông, phương tiện và đặc điểm tài xế.

## 🎯 Mục tiêu bài toán

- **Mục tiêu kinh doanh:** Thay thế cách ước lượng ETA tĩnh dựa trên khoảng cách và tốc độ trung bình bằng dự báo theo bối cảnh, hỗ trợ giảm hủy đơn và chi phí do giao trễ.
- **Bài toán ML:** Hồi quy có giám sát để dự đoán biến liên tục `time_taken` (phút).
- **Chỉ số đánh giá:** MAE, RMSE và $R^2$.

## 📊 Bộ dữ liệu & Tiền xử lý

- **Nguồn dữ liệu:** *Zomato Delivery Operations Analytics Dataset* (Kaggle), được dùng để mô phỏng hoạt động giao hàng của GoMeal.
- **Kích thước ban đầu:** 45.584 dòng × 20 cột.
- **Tiền xử lý và tạo đặc trưng:**
	- Làm sạch dữ liệu, xử lý giá trị khuyết thiếu và chuẩn hóa kiểu dữ liệu.
	- Giới hạn ngoại lai bằng Winsorization/Clipping ở ngưỡng 1% và 99%.
	- Trích xuất đặc trưng thời gian, mã hóa lượng giác sin/cos cho giờ, thứ và tháng.
	- Tính khoảng cách địa lý Haversine (`distance_km`), tạo biến kết hợp (`weather_traffic`, `traffic_city`) và biến lịch sử/trượt (`rolling_avg_city`).
	- Chia dữ liệu Train/Test theo tỷ lệ 80/20 theo trục thời gian để hạn chế rò rỉ dữ liệu.

## 🤖 Mô hình & kết quả

Ba mô hình được thử nghiệm và so sánh trên tập kiểm thử:

| Mô hình | MAE (phút) | RMSE (phút) | $R^2$ | Ghi chú |
| :--- | ---: | ---: | ---: | :--- |
| **Linear Regression** (Baseline) | 4.4036 | 5.5696 | 0.6519 | Mô hình cơ sở, dễ giải thích |
| **Random Forest** (Tuned) | 3.1242 | 3.8579 | 0.8330 | Bắt các quan hệ phi tuyến |
| **CatBoost Regression** (Tuned & Reduced) | **3.0805** | **3.8147** | **0.8367** | **Kết quả tốt nhất** |

### Phát hiện kinh doanh

1. **Bối cảnh vượt xa khoảng cách:** `weather_traffic` đóng góp 25,21% vào kết quả dự báo, cao hơn `distance_km` (19,06%).
2. **Năng lực tài xế và phương tiện:** Các yếu tố như tuổi, đánh giá tài xế và tình trạng xe chiếm 42,04% sức mạnh dự báo.

## 💻 Ứng dụng

Dự án có dashboard GoMeal xây dựng bằng **Streamlit**, hướng tới ba nhóm người dùng:

- **Khách hàng:** Theo dõi ETA theo điều kiện môi trường.
- **Tài xế:** Tham khảo lộ trình và cảnh báo về tình trạng phương tiện.
- **Bộ phận vận hành:** Điều phối tài xế và phân tích các kịch bản rủi ro.

Ứng dụng tải mô hình từ `app/catboost_reduced_model.cbm`. Hãy đặt file mô hình tại đúng đường dẫn này trước khi chạy dashboard; file đó chưa có trong repository hiện tại.

## 🚀 Cài đặt và chạy

Yêu cầu Python đã được cài đặt. Tạo môi trường ảo và cài các dependency:

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo, sau đó chạy:

```bash
python -m pip install -r requirements.txt
streamlit run app/app.py
```

Để mở các notebook phân tích:

```bash
jupyter notebook
```

## 📁 Cấu trúc thư mục

```text
GoMeal-Delivery-ETA-Prediction/
├── app/
│   └── app.py
├── data/
│   ├── processed/
│   │   ├── X_test.csv
│   │   ├── X_train.csv
│   │   ├── y_test.csv
│   │   └── y_train.csv
│   └── raw/
│       └── zomato_dataset.csv
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_model_linear_regression.ipynb
│   ├── 04_model_random_forest.ipynb.ipynb
│   ├── 05_model_catboost.ipynb
│   └── 06_insights_exploration.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```
