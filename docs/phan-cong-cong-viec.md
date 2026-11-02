# Bảng phân công công việc — Nhóm 4

Tổng: 25 phút thuyết trình + 8 phút Q&A · 1.655 dòng code · 54 test · 1 Docker image

---

## Bảng tổng hợp

| | Phần thuyết trình | Demo bằng | Sửa code gốc | Phát triển thêm | Phút |
|---|---|---|---|---|---|
| **P1** | Streamlit & Hạ tầng triển khai | Chế độ *P1 · Dữ liệu*, `docker compose up` | Ghép lại 4 khối thiếu của `myapp13`; gom 4–5 lần `yf.download` về 1 tầng | `cache.py` (cache 2 tầng), `Dockerfile` (multi-stage, non-root, healthcheck) | 5,5 |
| **P2** | Xây dựng danh mục đa tài sản | Chế độ *P2 · Portfolio* (tab Portfolio + Signals) | Align benchmark 1 lần; Signal Engine không tải lại dữ liệu từng mã | `costs.py` (phí + rebalance), `sizing.py` (tỷ trọng → phiếu lệnh) | 5,5 |
| **P3** | Đo lường rủi ro & nhân tố | Chế độ *P3 · Risk & Factor* (tab Risk + Factors) | Sửa momentum: `pct_change(12)` trên chuỗi lợi suất → `rolling(12).sum()` | `risk.py` (8 chỉ số đuôi), `factors.py` (OLS có t-stat, R²) | 6,0 |
| **P4** | Tối ưu hóa & chiến lược theo chu kỳ | Chế độ *P4 · Optimizer* (tab Optimize + Backtest) | Regime tính trên chuỗi **giá** (gốc tính trên lợi suất); `shift(1)` loại look-ahead | `optimizer.py` (+3 phương pháp, HRP), `regime.py` (`walk_forward`) | 6,0 |

Mở đầu 1,0 phút (P1) + chốt 1,0 phút (P4) = 25 phút.

---

## P1 — Streamlit & Hạ tầng triển khai

**Nội dung bài báo phụ trách:** TL;DR · Popularity of Streamlit (PyTrends) · Setting Up Conda Environment · Build Your First App · Personal Trading Analytics App · Ưu/nhược điểm Streamlit trong tài chính định lượng

**File bài gốc tham chiếu:** `myapp0.py`, `myapp1.py`

**Demo:** `docker compose up` live → sidebar chọn *P1 · Dữ liệu & Hạ tầng* → chỉ dòng "Nguồn: Cache trên đĩa (0 request mạng)" và thời gian tải (ms) → chạy `make size`

**Sửa trong code gốc**

| Vấn đề | Xử lý | File |
|---|---|---|
| `myapp13.py` bản in thiếu các khối định nghĩa `corr`, `rolling_vol`, `drawdown`, `rolling_beta`, `beta_latest`, `signal_df`, `final_signal`, `portfolio_total_return`, `alpha`, `regime_state` → NameError | Lấy ngược từ `myapp9`/`myapp10` ghép lại, phân về đúng module | toàn repo |
| `yf.download()` gọi 4–5 lần cho cùng khoảng thời gian | Gom về một tầng dữ liệu duy nhất | `core/data.py` (78 dòng) |
| Tab 5 dùng `col4` khi chỉ khai báo `st.columns(3)` | Khai báo đúng số cột | `tabs/tab_factors.py` |

**Phát triển thêm**

| # | Chức năng | File | Bằng chứng demo |
|---|---|---|---|
| 1 | Cache 2 tầng: RAM (`st.cache_data`) + đĩa (docker volume), có fallback khi mất mạng | `core/cache.py` (88 dòng) | Nhãn nguồn dữ liệu + số ms trên panel |
| 2 | Docker multi-stage, user non-root, `HEALTHCHECK` | `Dockerfile` | `make size` so kích thước image |
| (kèm) | Dữ liệu mẫu offline sinh bằng GBM, seed cố định | `core/fixtures.py` (109 dòng) | Toggle "Dùng dữ liệu mẫu" |

**Test:** `tests/test_p1_data.py` — 7 test (VIX nghịch biến SPY, cache key ổn định, dữ liệu tái lập 100%)

---

## P2 — Xây dựng danh mục đa tài sản

**Nội dung bài báo phụ trách:** Workflow 6 bước (User Input → Data → Portfolio Construction → Return & Performance → Visualizations → Signal) · Equal-weight portfolio · Metrics Total Return / Volatility / Sharpe · Interactive Portfolio App with User-Defined Weights · Signal Lab

**File bài gốc tham chiếu:** `myapp2.py`, `myapp5.py`

**Demo:** Chế độ *P2 · Portfolio Engine* → kéo slider tỷ trọng, Alpha đổi theo thời gian thực → bật phí 30bps + rebalance Tháng, đường "trước phí" tách ra → mở phiếu lệnh, đổi US sang HOSE

**Sửa trong code gốc**

| Vấn đề | Xử lý | File |
|---|---|---|
| Align portfolio với SPY lặp lại ở nhiều chỗ, mỗi file một kiểu | Align **một lần** duy nhất trong `build_portfolio` | `core/portfolio.py` |
| Signal Engine có vòng `for t in ticker_list: yf.download(t, ...)` — tải lại dữ liệu đã có | Dùng lại `md.prices`, toán học giữ nguyên | `core/portfolio.py:81` |
| `normalize_weights` không xử lý trường hợp tổng = 0 hoặc sai kích thước | Fallback về equal weight | `core/portfolio.py` |

**Giữ nguyên công thức bài báo:** Sharpe `mean/std*√252`, volatility `std*√252*100`, total return `(cum[-1]-1)*100`, alpha `port_return - spy_return`, tín hiệu MA20>MA50 trung bình các mã, ngưỡng 0,5

**Phát triển thêm**

| # | Chức năng | File | Con số demo |
|---|---|---|---|
| 1 | Chi phí giao dịch theo turnover + tái cân bằng định kỳ (Tháng/Quý/Năm) — bài gốc bỏ qua hoàn toàn | `core/costs.py` (77 dòng) | Hao hụt do phí hiện thành metric riêng |
| 2 | Tỷ trọng % → phiếu lệnh thực thi: số CP nguyên lô, tiền mặt dư, sai lệch làm tròn, xuất CSV | `core/sizing.py` (83 dòng) | HOSE lô 100, vốn 20.000$ → sai lệch 33%, không mua nổi 1 lô |

**Test:** `tests/test_p2_portfolio.py` — 19 test (phí cao hơn thì hao hụt lớn hơn; không rebalance thì turnover = 0; lô 100 lệch nhiều hơn lô 1)

---

## P3 — Đo lường rủi ro & nhân tố

**Nội dung bài báo phụ trách:** Correlation matrix · Drawdown · Enhanced Risk Metrics: Rolling Beta vs SPY (β>1 / β=1 / β<1 / β<0) · Công thức β = Cov/Var · Rolling Alpha · Momentum Exposure Factor Panel · Tab 5 Factors (SPY/QQQ/IWM/VIX beta, CAPM Alpha)

**File bài gốc tham chiếu:** `myapp7.py`, `myapp8.py`, khối Factor của `myapp13.py`

**Demo:** Chế độ *P3 · Risk & Factor* → bảng 8 chỉ số rủi ro → tab Factors, bảng OLS, chỉ cột `t_stat` và cột "Có ý nghĩa"

**Sửa trong code gốc**

| Vấn đề | Xử lý | File |
|---|---|---|
| `spy_momentum = spy_aligned.pct_change(12)` với `spy_aligned` là chuỗi **lợi suất** → vô nghĩa về kinh tế, phát nổ khi lợi suất đổi dấu | Dùng `benchmark_returns.rolling(12).sum()` = lợi suất tích lũy 12 phiên | `core/factors.py:93` |
| Phơi nhiễm nhân tố chỉ đo bằng tương quan trượt, không kiểm định | Bổ sung OLS có sai số chuẩn và t-stat | `core/factors.py` |

**Giữ nguyên công thức bài báo:** rolling beta `cov/var` cửa sổ 20, rolling alpha `pr.rolling(20).mean() - beta*br.rolling(20).mean()`, factor beta với SPY/QQQ/IWM/^VIX, CAPM alpha `ret_năm - beta*market_năm`

**Phát triển thêm**

| # | Chức năng | File | Chi tiết |
|---|---|---|---|
| 1 | 8 chỉ số đuôi rủi ro | `core/risk.py` (88 dòng) | VaR 95% (lịch sử + tham số), CVaR, Sortino, Calmar, Max DD, độ dài giai đoạn dưới đỉnh, skew, kurtosis |
| 2 | Hồi quy đa nhân tố OLS thuần numpy: hệ số, sai số chuẩn, t-stat, R² | `core/factors.py` (120 dòng) | Nhân tố long-short từ ETF: MKT, SIZE (IWM−SPY), GROWTH (QQQ−SPY), VOL; kèm `load_fama_french()` khi có mạng |
| (kèm) | Stress test theo beta | `core/risk.py` | Cú sốc thị trường −20% → tác động kỳ vọng |

**Test:** `tests/test_p3_risk.py` — 11 test (CVaR sâu hơn VaR; beta của chính benchmark = 1; OLS khớp hệ số đã biết; beta VIX âm)

---

## P4 — Tối ưu hóa & chiến lược theo chu kỳ

**Nội dung bài báo phụ trách:** Max Sharpe Optimization Engine (`portfolio_performance`, `neg_sharpe`, ràng buộc Σw=1 và 0≤w≤0,5, SLSQP) · Monte Carlo Efficient Frontier 5000 danh mục · Bull/Bear Regime Switching (score = trend + momentum − volatility) · Backtest Engine · Phần "What would a professional quant criticize?" · Conclusions

**File bài gốc tham chiếu:** `myapp9.py`, `myapp10.py`, `myapp13.py`

**Demo:** Chế độ *P4 · Optimizer & Regime* → bảng so sánh 5 phương pháp, chỉ cột "Max w %" → tab Backtest, walk-forward, chỉ delta Sharpe

**Sửa trong code gốc**

| Vấn đề | Xử lý | File |
|---|---|---|
| `spy_ma50 = spy_aligned.rolling(50).mean()` với `spy_aligned` là chuỗi **lợi suất** → so lợi suất hôm nay với lợi suất trung bình 50 phiên, không phải tín hiệu xu hướng | Tính trên chuỗi **giá**: `px.rolling(50).mean()` | `core/regime.py:26-28` |
| Chiến lược dùng regime **cùng ngày** để quyết định tỷ trọng ngày đó → look-ahead | `state.shift(1)` — hôm nay chỉ dùng thông tin đến hết hôm qua | `core/regime.py:60` |
| Bài toán tối ưu có thể vô nghiệm khi `max_weight < 1/n` | `max_w = max(max_weight, 1/n)` | `core/optimizer.py` |

**Giữ nguyên công thức bài báo:** `portfolio_performance` (r = w·μ, v = √(wᵀΣw), sharpe = r/v), SLSQP với `x0` equal weight, bounds (0; 0,5), Monte Carlo 5000 danh mục, ngưỡng regime ≥2 BULL / ≤0 BEAR

**Phát triển thêm**

| # | Chức năng | File | Con số demo |
|---|---|---|---|
| 1 | Thêm Min-Variance, Risk Parity, HRP (López de Prado) bên cạnh Max Sharpe + bảng so sánh | `core/optimizer.py` (147 dòng) | Max Sharpe dồn 50% (chạm trần) vs Risk Parity 41% |
| 2 | `walk_forward()`: tối ưu 252 phiên → áp dụng 63 phiên kế tiếp, lăn liên tục | `core/regime.py` (105 dòng) | Sharpe in-sample 1,36 vs out-of-sample 1,05 (−23%) |

**Test:** `tests/test_p4_optimizer.py` — 14 test (Min-Variance có vol thấp nhất; Monte Carlo không vượt nghiệm tối ưu; Risk Parity phân tán hơn Max Sharpe; in-sample luôn đẹp hơn OOS)

---

## Phần dùng chung

| Hạng mục | File | Ghi chú |
|---|---|---|
| Hợp đồng dữ liệu | `core/contracts.py` (151 dòng) | **FILE KHÓA** — chốt ngày 1, cần cả 4 người duyệt mới sửa |
| Điều phối UI | `app.py` (164 dòng) | Selector "Chế độ trình bày" để demo cá nhân |
| Quyền sở hữu file | `.github/CODEOWNERS` | Bản đồ ai viết file nào |
| CI | `.github/workflows/ci.yml` | Chạy 54 test + build Docker mỗi lần push |

## Lịch làm việc

| Mốc | Việc | Ai |
|---|---|---|
| Ngày 1 | Chốt `contracts.py`, commit `data/sample_*.csv` | Cả nhóm, P1 chủ trì |
| Ngày 2–5 | 4 nhánh song song, chỉ sửa file mình sở hữu | Song song |
| Ngày 6 | Merge PR, chạy test tích hợp, tập chuyển tiếp tab | Cả nhóm |
