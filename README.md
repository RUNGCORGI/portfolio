# Bloomberg-Style Portfolio Terminal & Quant Lab — Nhóm 4

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit 1.39](https://img.shields.io/badge/streamlit-1.39.0-FF4B4B.svg)](https://streamlit.io)
[![FastAPI 0.115+](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Docker Ready](https://img.shields.io/badge/docker-compose%20ready-2496ED.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-147%20passed-success.svg)](tests/)
[![Architecture](https://img.shields.io/badge/architecture-modular%20contracts-orange.svg)](core/contracts.py)

Nền tảng phân tích, quản trị danh mục đầu tư và giao dịch định lượng chuẩn tổ chức (Institutional Quant & Portfolio Management). Dự án tái cấu trúc toàn diện bài báo khoa học *"Building an Awesome Portfolio Management App in Streamlit from Scratch"* (Alexzap, 06/2026) từ mã nguồn thử nghiệm `myapp13.py` (800 dòng spaghetti, lỗi runtime, sai sót phương pháp luận) thành một hệ sinh thái định lượng công nghiệp đa nền tảng: **Streamlit Bloomberg Terminal**, **FastAPI Backend**, và **Web Dashboard** độc lập.

---

## ⚡ Khởi động nhanh (Quick Start)

### Cách 1: Docker (Khuyến nghị cho buổi thuyết trình / chấm điểm)
Chạy toàn bộ hệ thống bằng một lệnh duy nhất:

```bash
# Khởi chạy giao diện Terminal tại http://localhost:8501
docker compose up --build

# Chạy toàn bộ 147 test kiểm thử tự động
docker compose run --rm --profile ci tests
```

### Cách 2: Chạy trực tiếp với Python
Yêu cầu môi trường Python 3.10+ hoặc 3.11+:

```bash
# 1. Cài đặt thư viện
pip install -r requirements.txt

# 2. Sinh dữ liệu mẫu offline (chống lỗi kết nối mạng)
python -m core.fixtures

# 3. Khởi chạy Streamlit Terminal (Port 8501)
streamlit run app.py

# 4. (Tùy chọn) Khởi chạy FastAPI Backend & Web Dashboard (Port 8000)
python run_api.py
```

* Truy cập Terminal: **`http://localhost:8501`**
* Truy cập Web Dashboard & REST API: **`http://localhost:8000`** (Tài liệu Swagger: `http://localhost:8000/docs`)

---

## 🎯 Ba lỗi phương pháp luận nghiêm trọng của bài báo gốc đã khắc phục

1. **Look-ahead Bias (Gian lận tương lai):**
   * *Bài gốc:* Tối ưu hóa danh mục và kiểm thử hiệu suất (backtest) trên cùng một tập dữ liệu (In-sample overfitting), dẫn đến chỉ số Sharpe ảo tưởng lên tới 1.87.
   * *Khắc phục:* Nhóm xây dựng module Walk-Forward Validation (`core/regime.py` & `core/wf_backtest.py`), huấn luyện tối ưu trên 252 phiên và kiểm thử trượt trên 63 phiên kế tiếp độc lập (Out-of-sample). Đồng thời tích hợp kiểm định **Deflated Sharpe Ratio (DSR)** và **Probabilistic Sharpe Ratio (PSR)** để đo lường xác suất vượt trội do may mắn (`core/validation.py`).

2. **Tính toán Market Regime trên chuỗi dữ liệu sai:**
   * *Bài gốc:* Trong file `myapp10.py`, tác giả so sánh đường trung bình MA50 vs MA200 của chuỗi **lợi suất** SPY. Về mặt toán tài chính, tín hiệu xu hướng chu kỳ bắt buộc phải tính trên chuỗi **mức giá** (Price Level).
   * *Khắc phục:* Chuẩn hóa tín hiệu MA Crossover trên chuỗi giá tài sản, bổ sung mô hình **Gaussian Hidden Markov Model (HMM 3 trạng thái: Bull / Bear / Sideway)** học không giám sát phân phối lợi suất - biến động (`core/hmm_regime.py`).

3. **Tín hiệu Regime không có độ trễ thực thi (Signal Lag):**
   * *Bài gốc:* Ra quyết định tái cân bằng vào ngày $t$ dựa trên thông tin đóng cửa của chính ngày $t$, vi phạm thực tế thị trường.
   * *Khắc phục:* Áp dụng độ trễ chuẩn `state.shift(1)` — quyết định hôm nay chỉ được sử dụng dữ liệu đến hết ngày hôm trước. Bổ sung mô hình chi phí trượt giá (Slippage) và phí giao dịch (Transaction Cost BPS) trên mỗi vòng quay danh mục (Turnover) (`core/costs.py`).

---

## 🏛️ Kiến trúc hệ thống đa tầng (Architecture Overview)

Dự án áp dụng nguyên lý **Contract-Driven Architecture** (Kiến trúc hướng hợp đồng). Mọi module liên lạc thông qua hợp đồng dữ liệu chuẩn hóa bất biến `core/contracts.py`. Luồng phụ thuộc được kiểm soát nghiêm ngặt:

$$\text{Tầng Dữ liệu (P1)} \longrightarrow \text{Tầng Danh mục (P2)} \longrightarrow \left[ \text{Tầng Rủi ro (P3)} \parallel \text{Tầng Tối ưu hóa (P4)} \right] \longrightarrow \text{Quantamental AI \& Giao diện}$$

```text
portfolio/
├── app.py                      # Bộ điều phối Terminal Streamlit (10 Tabs hoàn chỉnh)
├── run_api.py                  # Điểm khởi chạy máy chủ FastAPI Backend
├── config.yaml                 # Cấu hình danh mục mặc định & tham số backtest
├── Dockerfile                  # Multi-stage build, Non-root user, Healthcheck
├── docker-compose.yml          # Điều phối container Streamlit & CI test runner
│
├── core/                       # ★ TRỌNG TÂM: Toàn bộ Engine Định lượng (Quant Core)
│   ├── contracts.py            # ★ FILE KHÓA — Hợp đồng cấu trúc dữ liệu chung
│   ├── data.py                 # Tải dữ liệu thị trường Mỹ (Yahoo Finance) & Multi-tier Cache
│   ├── vn_market.py            # Chế độ thị trường Việt Nam (VNINDEX, VN30, HOSE, vnstock)
│   ├── cache.py                # Cache 2 tầng: RAM (Streamlit) + Ổ đĩa (Parquet/Pickle)
│   ├── fixtures.py             # Sinh dữ liệu tổng hợp GBM offline có seed tái lập
│   ├── portfolio.py            # Tính toán danh mục, tỷ trọng, lợi suất tích lũy
│   ├── costs.py                # Mô hình chi phí giao dịch (turnover, commission bps)
│   ├── sizing.py               # Chuyển đổi % tỷ trọng thành Phiếu lệnh (Lot 100 HOSE, tiền dư)
│   ├── risk.py                 # 8 chỉ số đuôi rủi ro: Parametric/Historical VaR, CVaR, Drawdown
│   ├── factors.py              # Hồi quy OLS đa nhân tố (t-stat, p-value, R-squared)
│   ├── factor_lab.py           # Mô hình Carhart 4 nhân tố trực giao, kiểm tra VIF đa cộng tuyến
│   ├── optimizer.py            # Tối ưu hóa Markowitz: Max Sharpe, Min-Var, Risk Parity, HRP
│   ├── alloc_lab.py            # Tối ưu hóa nâng cao: Black-Litterman, Ledoit-Wolf Shrinkage
│   ├── regime.py               # Phân loại chu kỳ thị trường (Bull/Bear MA Crossover)
│   ├── hmm_regime.py           # Phân loại chu kỳ ẩn Gaussian Hidden Markov Model (HMM)
│   ├── wf_backtest.py          # Walk-forward backtest ngoài mẫu (Out-of-sample)
│   ├── validation.py           # Kiểm định xác suất: PSR, Deflated Sharpe Ratio, Bootstrap
│   ├── stress.py               # Stress test khủng hoảng (2008 GFC, 2020 COVID) & Monte Carlo
│   ├── xray.py                 # Portfolio X-Ray & Phân rã hiệu quả Brinson Attribution
│   ├── report.py               # Xuất báo cáo Tear Sheet HTML/PDF có mã băm tái lập SHA-256
│   ├── alerts.py               # Động cơ cảnh báo biến động & tích hợp Webhook (Slack/Discord)
│   └── journal.py              # Sổ nhật ký giao dịch mô phỏng (Paper Trading SQLite)
│
├── quantamental/               # ★ Phân tích kết hợp Quant + Fundamental + Sentiment Dual-RAG
│   ├── models.py               # Định nghĩa thực thể tin tức, BCTC, tín hiệu hợp nhất
│   ├── truth_table.py          # Ma trận chân lý 8 trạng thái (Bẫy tăng giá Bull Trap, FOMO, Vàng)
│   ├── rag_sentiment.py        # RAG Cảm xúc: NLTK VADER cho US, PhoBERT & PyVi cho VN
│   ├── rag_fundamental.py      # RAG BCTC: Đánh giá biên lợi nhuận, OCF, tỷ lệ nợ/VCSH
│   ├── news_engine.py          # Bộ thu thập tin tức thời gian thực & dự phòng offline
│   ├── financials_engine.py    # Phân tích chỉ số tài chính & dự phòng offline
│   ├── fixtures_cache.py       # Kho tri thức dữ liệu mẫu cho các mã VN30 & US Tech
│   └── tab_ui.py               # Giao diện trực quan hóa Quantamental AI & A/B Testing
│
├── api/                        # ★ Tầng dịch vụ REST API (FastAPI)
│   ├── main.py                 # Khởi tạo FastAPI app, middleware GZip, timing, static files
│   ├── schemas.py              # Pydantic schemas cho Request & Response
│   ├── services.py             # Cầu nối gọi logic tính toán từ core/ sang JSON
│   └── routers/
│       ├── analytics.py        # Endpoints phân tích: danh mục, rủi ro, tối ưu, quantamental
│       └── monitor.py          # Endpoints cảnh báo, webhook, nhật ký giao dịch
│
├── web/                        # ★ Giao diện Web Dashboard độc lập (HTML5/CSS3/Vanilla JS)
│   ├── index.html              # Trang chủ Dashboard điều hành
│   ├── css/style.css           # Giao diện Bloomberg Dark Mode hiện đại
│   └── js/
│       ├── api.js              # Client tương tác với FastAPI
│       ├── charts.js           # Vẽ biểu đồ tương tác với Chart.js
│       └── app.js              # Xử lý sự kiện giao diện
│
├── tabs/                       # Các Tab giao diện Streamlit (phân tách quyền sở hữu)
│   ├── tab_portfolio.py        # Tab 1: Cấu trúc danh mục & Hiệu suất
│   ├── tab_signals.py          # Tab 2: Tín hiệu kỹ thuật MA Crossover
│   ├── tab_risk.py             # Tab 3: Phân tích rủi ro & Đo lường đuôi phân phối
│   ├── tab_factors.py          # Tab 4: Phân tích nhân tố thị trường (OLS)
│   ├── tab_optimize.py         # Tab 5: Tối ưu hóa phân bổ tỷ trọng
│   ├── tab_backtest.py         # Tab 6: Kiểm thử chiến lược theo chu kỳ
│   ├── tab_quantamental.py     # Tab 7: Quantamental AI & Dual-RAG
│   ├── tab_xray.py             # Tab 8: Portfolio X-Ray & Báo cáo Tear Sheet
│   ├── tab_monitor.py          # Tab 9: Giám sát thời gian thực & Nhật ký lệnh
│   └── tab_lab.py              # Tab 10: Quant Lab nâng cao (HMM, Black-Litterman, Stress)
│
├── tests/                      # ★ Hệ thống kiểm thử tự động toàn diện (147 test cases)
│   ├── test_p1_data.py         # 7 tests: Tầng dữ liệu & Cache
│   ├── test_p2_portfolio.py    # 19 tests: Tầng Danh mục, Chi phí, Sizing phiếu lệnh
│   ├── test_p3_risk.py         # 11 tests: Tầng Rủi ro & Hồi quy nhân tố
│   ├── test_p4_optimizer.py    # 13 tests: Tối ưu hóa & Walk-Forward
│   ├── test_quantamental.py    # 16 tests: Ma trận chân lý, Dual-RAG, A/B Testing, API
│   ├── test_upgrade_pack.py    # 25 tests: Quant Lab (Alloc, HMM, Stress, Validation)
│   └── test_p1_p3.py           # 47 tests: Factor Lab, X-Ray, Báo cáo, Thị trường VN, Journal
│
└── docs/                       # Tài liệu hướng dẫn chi tiết
    ├── huong-dan-chay.md       # Hướng dẫn cài đặt & khắc phục sự cố chi tiết
    ├── huong-dan-nang-cap.md   # Thiết kế gói nâng cấp Quant Lab
    ├── kich-ban-demo.md        # Kịch bản thuyết trình 25 phút phân vai
    ├── phan-1-va-3.md          # Chi tiết kỹ thuật Factor Lab & Portfolio X-Ray
    └── phan-cong-cong-viec.md  # Ma trận phân công chi tiết & cam kết điểm số
```

---

## 👥 Phân công công việc & Bằng chứng đóng góp cá nhân

Mỗi thành viên sở hữu branch riêng, file mã nguồn riêng, bộ kiểm thử độc lập và tab trình diễn cá nhân trên Sidebar:

| Vai trò | Thành viên phụ trách | Module sở hữu | Bộ kiểm thử (`pytest`) | Phần phát triển nâng cao (Vượt trội bài báo gốc) |
|:---|:---|:---|:---|:---|
| **P1** | *Thành viên 1* | `core/data.py`<br>`core/cache.py`<br>`core/fixtures.py`<br>`core/vn_market.py`<br>`Dockerfile` | `tests/test_p1_data.py`<br>(7 tests) | **(1)** Cache 2 tầng (RAM + Parquet Disk) triệt tiêu 100% request trùng.<br>**(2)** Tích hợp thị trường Việt Nam (VN30, VNINDEX).<br>**(3)** Docker multi-stage, non-root user, healthcheck tự động.<br>**(4)** Bộ sinh dữ liệu mẫu Geometric Brownian Motion (GBM) offline 100%. |
| **P2** | *Thành viên 2* | `core/portfolio.py`<br>`core/costs.py`<br>`core/sizing.py`<br>`tabs/tab_portfolio.py`<br>`tabs/tab_signals.py` | `tests/test_p2_portfolio.py`<br>(19 tests) | **(1)** Mô hình chi phí giao dịch theo biến động vòng quay danh mục (Turnover BPS) và chu kỳ tái cân bằng định kỳ.<br>**(2)** Bộ máy phân bổ phiếu lệnh thực thi: Quy đổi tỷ trọng % sang số cổ phiếu nguyên lô (Lot 100 HOSE hoặc cổ phiếu lẻ US), tiền mặt thặng dư, sai lệch thực tế. Xuất phiếu lệnh CSV. |
| **P3** | *Thành viên 3* | `core/risk.py`<br>`core/factors.py`<br>`core/factor_lab.py`<br>`tabs/tab_risk.py`<br>`tabs/tab_factors.py` | `tests/test_p3_risk.py`<br>(11 tests) | **(1)** Bộ 8 chỉ số đuôi rủi ro phân phối phi chuẩn: Historical/Parametric VaR (95%, 99%), CVaR, Sortino, Calmar, Max Drawdown Duration, Skewness, Kurtosis.<br>**(2)** Factor Lab: Hồi quy OLS đa nhân tố Carhart 4-factor; trực giao hóa các nhân tố so với thị trường; kiểm tra đa cộng tuyến VIF; phân tích trôi dạt phong cách (Style Drift). |
| **P4** | *Thành viên 4* | `core/optimizer.py`<br>`core/regime.py`<br>`core/alloc_lab.py`<br>`core/hmm_regime.py`<br>`tabs/tab_optimize.py`<br>`tabs/tab_backtest.py` | `tests/test_p4_optimizer.py`<br>(13 tests) | **(1)** Mở rộng 6 thuật toán phân bổ danh mục: Markowitz Max Sharpe, Minimum Variance, Inverse Volatility, Risk Parity (Equal Risk Contribution), Hierarchical Risk Parity (HRP), Black-Litterman.<br>**(2)** Walk-Forward Validation bóc tách Look-ahead bias.<br>**(3)** Mô hình nhận diện chu kỳ ẩn Gaussian Hidden Markov Model (HMM). |
| **P5 / AI** | *Thành viên 5 / Toàn đội* | `quantamental/*`<br>`core/xray.py`<br>`core/stress.py`<br>`core/report.py`<br>`api/*`<br>`web/*` | `tests/test_quantamental.py`<br>`tests/test_upgrade_pack.py`<br>`tests/test_p1_p3.py`<br>(97 tests) | **(1)** Quantamental AI & Dual-RAG: Kết hợp Kỹ thuật (T) + Cảm xúc tin tức (S) + BCTC (F) cùng ma trận chân lý 8 trạng thái triệt tiêu bẫy tăng giá (Bull Trap).<br>**(2)** Portfolio X-Ray & Brinson Performance Attribution.<br>**(3)** Stress Testing khủng hoảng lịch sử & 1.000 kịch bản Monte Carlo.<br>**(4)** Xuất báo cáo Tear Sheet HTML/PDF có băm SHA-256.<br>**(5)** Hệ thống REST API (FastAPI) & Web Dashboard độc lập. |

---

## 🔬 Chi tiết các năng lực định lượng cốt lõi

### 1. Quantamental AI & Dual-RAG Engine
* **Phân tích đa chiều:**
  * **Flag T (Technical):** Tín hiệu động lượng kỹ thuật từ giao cắt MA ngắn hạn và dài hạn.
  * **Flag S (Sentiment RAG):** Khai phá dữ liệu báo chí thời gian thực. Tích hợp từ điển tài chính NLTK VADER cho thị trường Mỹ và mô hình ngôn ngữ **PhoBERT** cho thị trường chứng khoán Việt Nam.
  * **Flag F (Fundamental RAG):** Bóc tách báo cáo tài chính: Biên lợi nhuận ròng, dòng tiền hoạt động kinh doanh (OCF), tăng trưởng doanh thu YoY, và đòn bẩy nợ/VCSH.
* **Ma trận chân lý 8 trạng thái (Truth Table):** Nhận diện chính xác mẫu hình thị trường:
  * `[1, 1, 1]`: **Tín hiệu Vàng** — Hội tụ đủ kỹ thuật, truyền thông và nội tại doanh nghiệp.
  * `[1, 1, 0]`: **Bẫy tăng giá (Bull Trap / FOMO)** — Giá tăng và tin tức thổi phồng nhưng nội tại suy yếu $\rightarrow$ Khuyến nghị giảm tỷ trọng/đứng ngoài.
  * `[1, 0, 1]`: **Ngọc trong đá (Hidden Gem)** — Nền tảng vững chắc, kỹ thuật tốt dù truyền thông chưa chú ý $\rightarrow$ Khuyến nghị gom dần.
  * `[0, 0, 0]`: **Rủi ro suy thoái** $\rightarrow$ Kích hoạt vị thế phòng thủ.
* **Thực nghiệm đối chứng A/B Testing:** So sánh trực quan hiệu suất lũy kế và chỉ số Sharpe giữa danh mục truyền thống và danh mục tối ưu hóa bởi Quantamental AI.

### 2. Mô hình Nhân tố & Factor Lab
* Khắc phục triệt để việc hồi quy đơn biến từng ETF thô dẫn đến sai lệch đa cộng tuyến nghiêm trọng (VIF > 12 ở bài báo gốc).
* Xây dựng các nhân tố trực giao hóa dạng Long/Short so với thị trường:
  $$\text{GROWTH} = \text{QQQ} - \text{SPY}, \quad \text{SIZE} = \text{IWM} - \text{SPY}, \quad \text{GOLD} = \text{GLD} - \text{SPY}$$
* Giảm hệ số phóng đại phương sai **VIF từ 12.50 xuống dưới 1.15**, đảm bảo hệ số hồi quy $t$-stat và $p$-value có ý nghĩa thống kê thực sự.

### 3. Tối ưu hóa phân bổ tỷ trọng & Quant Lab
* **Hierarchical Risk Parity (HRP):** Sử dụng thuật toán phân cụm phân cấp (Hierarchical Clustering) dựa trên ma trận khoảng cách tương quan, phân bổ rủi ro tối ưu mà không cần nghịch đảo ma trận hiệp phương sai (tránh rủi ro ma trận suy biến).
* **Ledoit-Wolf Shrinkage:** Co ngót ma trận hiệp phương sai mẫu về cấu trúc không đổi, triệt tiêu nhiễu mẫu nhỏ.
* **Black-Litterman Model:** Kết hợp trạng thái cân bằng thị trường (Impled Equilibrium Returns) với các quan điểm cá nhân (Investor Views) để tạo ra phân bổ ổn định.

### 4. Kiểm định rủi ro, Stress Testing & Kiểm tra may mắn
* **Stress Test kịch bản lịch sử:** Tái hiện tác động của Khủng hoảng tài chính toàn cầu 2008 (GFC), Cú sốc sụp đổ COVID-19 (03/2020), và Bong bóng Dot-Com 2000 lên danh mục hiện tại.
* **Mô phỏng Monte Carlo:** 1.000 đường đi mô phỏng theo phân phối hình học Brown để ước lượng tổn thất tối đa kỳ vọng.
* **Deflated Sharpe Ratio (DSR):** Điều chỉnh chỉ số Sharpe theo số lượng chiến lược đã thử nghiệm, độ lệch (Skewness) và độ nhọn (Kurtosis) của chuỗi lợi suất để loại bỏ hiện tượng "Data Snooping".

### 5. Thực thi lệnh thực tế & Thị trường Việt Nam
* **Cơ chế khớp lệnh HOSE:** Tự động quy đổi tỷ trọng tối ưu sang số lượng cổ phiếu tròn lô 100 theo quy định Sở GDCK TP.HCM, tự động giữ lại lượng tiền mặt lẻ (Cash Buffer) và tính toán sai lệch bám đuổi danh mục (Tracking Error due to Rounding).
* **Bảng theo dõi thị trường Việt Nam:** Hỗ trợ đầy đủ dữ liệu nhóm cổ phiếu VN30 (VCB, FPT, HPG, VIC, VNM, MSN, TCB, MWG...) với benchmark chuẩn VNINDEX.

---

## 🌐 FastAPI REST API & Web Dashboard

Hệ thống cung cấp kiến trúc phân tầng chuyên nghiệp: Toàn bộ thuật toán được đóng gói trong `core/`, cung cấp API chuẩn mực qua FastAPI và hiển thị qua giao diện Web nhẹ nhàng:

* **Swagger API Documentation:** `http://localhost:8000/docs`
* **Healthcheck:** `GET /health`
* **Endpoints phân tích danh mục:** `POST /api/analytics/portfolio`, `POST /api/analytics/risk`, `POST /api/analytics/optimize`
* **Endpoints Quantamental AI:** `POST /api/quantamental`
* **Endpoints giám sát & cảnh báo:** `GET /api/monitor/alerts`, `POST /api/monitor/webhook`
* **Giao diện Web Client:** Phục vụ trực tiếp tại root `http://localhost:8000` với giao diện Bloomberg Dark Mode, biểu đồ Chart.js mượt mà, thời gian phản hồi máy chủ được ghi nhận trực tiếp trên HTTP Header `X-Process-Time-Ms`.

---

## 🧪 Hệ thống kiểm thử tự động (Test Suite)

Dự án có **147 ca kiểm thử tự động** được tổ chức chặt chẽ với `pytest`, đảm bảo mọi tính toán toán học và luồng API hoạt động chính xác:

```bash
# Chạy toàn bộ test suite
pytest -v

# Chạy riêng theo từng phân hệ
pytest tests/test_p1_data.py          # Kiểm tra Cache, GBM, Fallback
pytest tests/test_p2_portfolio.py     # Kiểm tra Lợi suất, Chi phí, Sizing lô 100
pytest tests/test_p3_risk.py          # Kiểm tra 8 chỉ số rủi ro, OLS
pytest tests/test_p4_optimizer.py     # Kiểm tra Markowitz, HRP, Walk-forward
pytest tests/test_upgrade_pack.py     # Kiểm tra Quant Lab, HMM, Stress test
pytest tests/test_p1_p3.py            # Kiểm tra Factor Lab, X-Ray, Báo cáo, VN Market
pytest tests/test_quantamental.py     # Kiểm tra Dual-RAG, Ma trận chân lý, A/B Testing
```

---

## 🎭 Kịch bản Demo 25 phút thuyết trình

Sidebar của Terminal tích hợp sẵn thanh chọn **"Chế độ trình bày"**, cho phép từng thành viên bật/tắt chính xác các tab thuộc trách nhiệm của mình:

| Thời lượng | Người trình bày | Chế độ Sidebar | Thao tác demo trên màn hình | Thông điệp cốt lõi |
|:---|:---|:---|:---|:---|
| **00:00 – 01:30** | **P1** | Slide kiến trúc | Giới thiệu bài báo gốc, 3 lỗi phương pháp luận nghiêm trọng và kiến trúc hợp đồng module hóa. | *"Bài báo dừng ở file 800 dòng lỗi thời; nhóm chúng em bắt đầu từ việc sửa lỗi và chuẩn hóa công nghiệp."* |
| **01:30 – 06:30** | **P1** | `P1 · Dữ liệu & Hạ tầng` | Khởi chạy Docker live; chứng minh Cache 2 tầng "0 request mạng" và đo ms; bật toggle dữ liệu mẫu offline; giới thiệu chế độ VN Market. | *"Docker và Cache không chỉ làm app nhanh hơn, mà làm quy trình nghiên cứu tài chính có thể tái lập 100%."* |
| **06:30 – 11:30** | **P2** | `P2 · Portfolio Engine` | Kéo slider tỷ trọng $\rightarrow$ Lợi suất & Alpha thay đổi; bật phí 30 bps + tái cân bằng định kỳ; chuyển sang Phiếu lệnh HOSE (lô 100, tiền lẻ). | *"Bài gốc báo lãi 576% mà chưa trừ đồng phí nào; và chưa nhà đầu tư nào có thể đặt lệnh mua 33.3% cổ phiếu."* |
| **11:30 – 16:30** | **P3** | `P3 · Risk & Factor` | Trình chiếu bảng 8 chỉ số đuôi rủi ro (VaR/CVaR); mở Factor Lab chứng minh VIF giảm từ 12.5 về 1.1; kiểm định $t$-stat có ý nghĩa. | *"Phơi nhiễm rủi ro chỉ có ý nghĩa khi kiểm định \|t\| > 2; bài gốc hoàn toàn bỏ qua suy diễn thống kê."* |
| **16:30 – 21:30** | **P4** | `P4 · Optimizer & Regime` | So sánh 5 phương pháp tối ưu (Markowitz vs HRP); chứng minh Out-of-sample Sharpe suy giảm qua Walk-Forward; trực quan hóa Gaussian HMM 3 trạng thái. | *"Chỉ số Sharpe 1.87 của bài gốc là ảo tưởng In-sample. Walk-forward bóc tách chính xác độ trôi dạt thực tế."* |
| **21:30 – 24:00** | **P5 / AI** | `🧠 Quantamental AI` | Trình chiếu ma trận chân lý 8 trạng thái; minh họa phát hiện Bull Trap khi tin tốt mà BCTC xấu; thực nghiệm đối chứng A/B Testing; mở nhanh Web Dashboard & Swagger API. | *"AI kết hợp định lượng giúp nhà đầu tư miễn nhiễm với bẫy FOMO và tin tức truyền thông sai lệch."* |
| **24:00 – 25:00** | **Cả nhóm** | Terminal / CI | Chạy lệnh `docker compose run --rm --profile ci tests` $\rightarrow$ 147 test xanh đồng loạt. | *"Toàn bộ hệ thống được bảo chứng bằng kiểm thử tự động toàn diện."* |

---

## 🔒 Phương án dự phòng sự cố phòng học

1. **Mất kết nối Internet:** Hệ thống tự động chuyển sang chế độ dự phòng Offline với dữ liệu GBM và kho tri thức tài chính được nạp sẵn trong `core/fixtures.py` và `quantamental/fixtures_cache.py`.
2. **Máy phòng học không hỗ trợ Docker:** Chạy trực tiếp `streamlit run app.py` hoặc `python run_api.py` bằng Python cục bộ.
3. **Màn hình chiếu gặp sự cố:** Mỗi thành viên đều chuẩn bị sẵn bản video quay màn hình thao tác chuẩn dài 60 giây và file xuất Báo cáo PDF Tear Sheet chất lượng cao.

---

## 📚 Tài liệu tham khảo & Nguồn học thuật

1. **Alexzap** (06/2026) — *Building an Awesome Portfolio Management App in Streamlit from Scratch*.
2. **Markowitz, H.** (1952) — *Portfolio Selection*. The Journal of Finance, 7(1), 77-91.
3. **Fama, E. F., & French, K. R.** (1993) — *Common risk factors in the returns on stocks and bonds*. Journal of Financial Economics, 33(1), 3-56.
4. **Carhart, M. M.** (1997) — *On Persistence in Mutual Fund Performance*. The Journal of Finance, 52(1), 57-82.
5. **Lopez de Prado, M.** (2016) — *Building Diversified Portfolios that Outperform Out-of-Sample*. The Journal of Portfolio Management, 42(4), 59-69. (Hierarchical Risk Parity).
6. **Bailey, D. H., & Lopez de Prado, M.** (2014) — *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality*. Journal of Portfolio Management.
7. **Black, F., & Litterman, R.** (1992) — *Global Portfolio Optimization*. Financial Analysts Journal, 48(5), 28-43.
8. **Nguyen et al.** (2020) — *PhoBERT: Pre-trained language models for Vietnamese*. Findings of EMNLP 2020.
