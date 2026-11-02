# Phần 1 và Phần 3 — tài liệu bổ sung

Tài liệu này mô tả phần mở rộng so với gói nâng cấp đợt đầu. Nguyên tắc vẫn giữ nguyên: **chỉ thêm file mới, không sửa bất kỳ file nào trong dự án của bạn**.

## File mới

```text
core/
├── factor_lab.py     ← Phần 1: mô hình nhân tố đầy đủ
├── xray.py           ← Phần 3: Portfolio X-Ray
├── report.py         ← Phần 3: xuất HTML/PDF có mã tái lập
├── alerts.py         ← Phần 3: động cơ cảnh báo + webhook
├── vn_market.py      ← Phần 3: chế độ thị trường Việt Nam
└── journal.py        ← Phần 3: nhật ký giao dịch mô phỏng (SQLite)

tabs/
├── tab_xray.py       ← MỚI (tên không trùng tab_factors.py của bạn)
└── tab_monitor.py    ← MỚI

tests/
└── test_p1_p3.py     ← 47 test mới
```

Tổng cộng **77 test** đều pass. Không cần cài thêm thư viện: `reportlab`, `weasyprint`, `sqlite3`, `requests` đều có sẵn, và module nào thiếu cũng có đường lui dự phòng.

Tích hợp vào `app.py`:

```python
from tabs import tab_xray, tab_monitor

tabs = st.tabs([..., "X-Ray & Báo cáo", "Giám sát"])
with tabs[-2]:
    tab_xray.render()
with tabs[-1]:
    tab_monitor.render()
```

---

## Phần 1 — Mô hình nhân tố (`factor_lab.py`)

### Vấn đề của cách làm cũ

Bài gốc hồi quy danh mục với SPY, QQQ, IWM, VIX **từng cái một**. Bốn chỉ số này tương quan rất mạnh, nên mỗi hệ số đều đang đếm lại cùng một rủi ro thị trường. Hàm `naive_vs_orthogonal()` chứng minh bằng số:

| Cách dựng | Nhân tố | VIF | Đánh giá |
|---|---|---|---|
| Dùng ETF thô (bài gốc) | SPY | **12,50** | Chồng lấn nghiêm trọng |
| Dùng ETF thô (bài gốc) | QQQ | 9,30 | Cần lưu ý |
| Long/short đã trừ SPY | GROWTH | **1,13** | Chấp nhận được |
| Long/short đã trừ SPY | SIZE | **1,03** | Chấp nhận được |

VIF trên 10 nghĩa là hệ số của nhân tố đó về cơ bản không đọc được.

### Bốn cải tiến

**1. Nhân tố long/short.** `GROWTH = QQQ − SPY`, `SIZE = IWM − SPY`, `GOLD = GLD − SPY`, `DURATION = TLT − SPY`. Trừ đi thị trường chung làm các nhân tố gần trực giao.

**2. Động lượng tự dựng.** `build_momentum_factor()` xếp hạng chính rổ cổ phiếu của bạn theo quy ước 12−1 (bỏ qua tháng gần nhất để tránh hiệu ứng đảo chiều ngắn hạn), mua nhóm dẫn đầu bán nhóm tụt hậu. Bài gốc chỉ lấy `SPY.pct_change(12)` — thứ đó thực chất là lợi suất thị trường trễ, không phải động lượng.

**3. Sai số chuẩn Newey–West.** Lợi suất có tự tương quan và phương sai thay đổi; sai số OLS thông thường quá nhỏ nên thống kê t bị thổi phồng. Hàm `compare_ols_vs_hac()` cho thấy trên dữ liệu thử nghiệm, nhân tố động lượng có t = 16,92 theo OLS nhưng chỉ **11,39** theo Newey–West — **thổi phồng 1,49 lần**. Đây đúng là loại sai lầm khiến người ta kết luận nhầm là "có alpha".

**4. Phân rã lợi nhuận.** Kết quả chạy thử với danh mục chia đều:

| Nguồn | Tỷ trọng |
|---|---|
| Alpha (kỹ năng) | 40,1% |
| Nhân tố MKT | 46,0% |
| Nhân tố MOM | 7,9% |
| Các nhân tố khác | 6,0% |

Gần một nửa lợi nhuận chỉ đến từ việc nắm giữ thị trường. Đây là câu trả lời thẳng cho câu hỏi "giỏi hay chỉ là gánh rủi ro".

Ngoài ra `rolling_exposures()` và `style_drift_score()` phát hiện hiện tượng **trôi phong cách** — danh mục khởi đầu trung lập nhưng dần nghiêng mạnh sang công nghệ.

---

## Phần 3 — Năm tính năng demo

### 1. Portfolio X-Ray (`xray.py`)

Tearsheet 18 chỉ số, trong đó ba chỉ số ít gặp nhưng rất đáng giá:

- **Chỉ số Ulcer** — đo cả độ sâu lẫn độ dài sụt giảm. Hai danh mục cùng giảm 30% nhưng một cái hồi phục sau hai tháng, cái kia mất hai năm: max drawdown không phân biệt được, Ulcer thì có.
- **Tỷ lệ đuôi** — so sánh đuôi lãi với đuôi lỗ.
- **Active share** — chỉ số quỹ dùng để chứng minh không phải "quỹ chỉ số trá hình".

**Phân rã rủi ro** là phần đắt nhất. Kết quả chạy thử với HRP:

| Mã | Trọng số vốn | Tỷ trọng rủi ro | Chênh lệch |
|---|---|---|---|
| KO | 41,0% | 33,4% | −7,6% |
| NVDA | **4,3%** | **9,4%** | **+5,2%** |

NVDA chỉ chiếm 4,3% vốn nhưng gánh 9,4% rủi ro. Biểu đồ tròn phân bổ vốn không bao giờ cho thấy điều này.

Thêm `benchmark_suite()` cung cấp ba mốc bắt buộc: chia đều, thị trường, và 60/40 xấp xỉ.

### 2. Xuất báo cáo (`report.py`)

Sinh HTML một file (ảnh nhúng base64) và PDF. Mỗi báo cáo mang **mã tái lập** — chuỗi băm SHA-256 của toàn bộ cấu hình, danh sách mã, khoảng thời gian và số phiên dữ liệu.

Ba tính chất đã được kiểm thử:
- Cùng cấu hình luôn cho cùng mã băm.
- Thứ tự khai báo mã không ảnh hưởng tới mã băm.
- Đổi một tham số là mã băm đổi theo.

Đồ thị vẽ bằng Matplotlib chứ không phải Plotly, vì Plotly cần Chrome/Kaleido để xuất ảnh — Matplotlib luôn chạy được trong Docker và trên máy chấm bài. PDF ưu tiên WeasyPrint, không được thì tự động lui về ReportLab.

### 3. Động cơ cảnh báo (`alerts.py`)

Bảy luật khai báo được: sụt giảm, biến động, beta, tập trung, VaR, chuyển chế độ, và tương quan tăng cao.

Luật `REGIME` chỉ kích hoạt **khi chuyển trạng thái**, không phải mỗi ngày thị trường đang ở chế độ BEAR — chi tiết nhỏ nhưng quyết định việc cảnh báo có dùng được hay chỉ gây nhiễu. Đã có test riêng cho điều này.

Gửi webhook tới Teams, Slack hoặc Telegram. **Mọi lỗi mạng đều được bắt**, không ném ngoại lệ: dashboard không bao giờ được sập chỉ vì webhook chết. Có chế độ xem trước để demo mà không cần cấu hình webhook thật. Nhật ký lưu SQLite.

### 4. Thị trường Việt Nam (`vn_market.py`)

Chứng minh lớp trừu tượng dữ liệu hoạt động: đổi sang cổ phiếu Việt Nam mà **không sửa một dòng nào** trong các module phân tích.

Ba đặc thù đã xử lý:

- **Biên độ dao động theo sàn** (HOSE ±7%, HNX ±10%, UPCoM ±15%). Chi tiết kỹ thuật đáng nói: biên độ áp lên lợi suất *giản đơn*, không phải log-return. Phiên bản đầu của mình cắt nhầm trên log-return khiến `exp(0,07) − 1 = 7,25%` vượt biên độ — chính test `test_vn_prices_respect_limits` đã bắt được lỗi này.
- **Chi phí bất đối xứng**: thuế thu nhập cá nhân 0,1% chỉ đánh một chiều khi bán, nên chi phí mua (0,25%) khác chi phí bán (0,35%). Mô hình chi phí chuẩn Mỹ tính sai điều này.
- **Rủi ro tập trung ngành**: nhóm ngân hàng, bất động sản và chứng khoán chi phối phần lớn chỉ số VNINDEX.

`price_limit_hits()` đếm số phiên chạm trần/sàn — chạm nhiều cảnh báo backtest có thể lạc quan, vì khi cổ phiếu nằm sàn thì thường không khớp được lệnh bán.

Provider thử lần lượt: `vnstock` nếu đã cài → CSV cục bộ → mô phỏng. Demo không bao giờ chết vì mất mạng.

### 5. Nhật ký giao dịch (`journal.py`)

Backtest giả định bạn luôn thực hiện đúng tín hiệu. Thực tế con người do dự, vào lệnh trễ, bỏ qua tín hiệu vì sợ.

Sổ này ghi quyết định thật vào SQLite, tính giá vốn bình quân gia quyền, tách lãi lỗ đã thực hiện khỏi chưa thực hiện, và đo hai thứ:

- **Tỷ lệ tuân thủ** — bao nhiêu phần trăm lệnh làm theo mô hình.
- **Khoảng cách thực thi** — chênh lệch giữa NAV thực tế và NAV mô hình.

Giá trị âm nghĩa là khâu thực thi đang bào mòn lợi nhuận. Đây là con số **không backtest nào tính được**, và thường hé lộ điều bất ngờ: nhiều người bỏ qua tín hiệu đúng lúc nó có giá trị nhất, tức là khi thị trường đang hoảng loạn.

---

## Phân công cập nhật cho nhóm bốn người

| Người | Sở hữu | Nội dung trình bày |
|---|---|---|
| 1 | `upgrade_api.py`, `vn_market.py`, cache | Kiến trúc dữ liệu nhiều nguồn, chế độ ngoại tuyến, đặc thù thị trường Việt Nam |
| 2 | `alloc_lab.py`, `factor_lab.py` | Vì sao Max Sharpe bất ổn; đa cộng tuyến và sai số chuẩn HAC |
| 3 | `wf_backtest.py`, `validation.py`, `xray.py` | Rò rỉ dữ liệu tương lai, chi phí, kỹ năng hay may mắn, phân rã rủi ro |
| 4 | `hmm_regime.py`, `stress.py`, `alerts.py`, `journal.py`, các tab | Mô hình chế độ xác suất, phòng kịch bản, giám sát và khoảng cách thực thi |

## Câu hỏi hội đồng hay hỏi

**"Vì sao phải dùng Newey–West?"**
Vì lợi suất có tự tương quan. Bảng `compare_ols_vs_hac` cho thấy thống kê t của nhân tố động lượng bị thổi phồng 1,49 lần nếu dùng OLS thường — đủ để biến một kết quả vô nghĩa thành "có ý nghĩa ở mức 1%".

**"Alpha 40% nghe cao quá, có thật không?"**
Đó là tỷ trọng trong phân rã, không phải lợi suất. Và nó được tính trên dữ liệu mô phỏng ngoại tuyến. Quan trọng hơn: `validation.py` đã chỉ ra Deflated Sharpe chỉ 0,54 với 20 lần thử, tức là nhóm không khẳng định alpha này có thật.

**"Mã tái lập để làm gì?"**
Để người chấm có thể xác minh hai báo cáo có đến từ cùng một lần chạy hay không. Ảnh chụp màn hình dashboard không chứng minh được bạn chạy với tham số nào.

**"Nhật ký giao dịch có phải thừa không?"**
Nó đo thứ duy nhất mà backtest không đo được: khoảng cách giữa tín hiệu và hành vi thực tế của con người.
