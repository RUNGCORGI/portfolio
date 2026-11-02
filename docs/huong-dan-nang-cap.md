# Gói nâng cấp Quant Lab — hướng dẫn tích hợp

## Nguyên tắc thiết kế

Gói này được viết **mà không cần đọc code gốc của bạn**, nên nó tuân thủ ba quy tắc:

1. **Chỉ thêm, không sửa.** Không có file nào trong dự án của bạn bị thay đổi.
2. **Tên file không trùng.** `alloc_lab.py` chứ không phải `optimizer.py`; `wf_backtest.py` chứ không phải sửa `tab_backtest.py`; `hmm_regime.py` chứ không phải `regime.py`.
3. **Chỉ phụ thuộc vào kiểu dữ liệu phổ quát.** Mọi hàm nhận `pd.DataFrame` giá hoặc lợi suất, không gọi hàm nội bộ của bạn. Nhờ vậy dù `core/data.py` của bạn đặt tên hàm thế nào, gói vẫn chạy.

Nếu dự án của bạn có `core/data.py` với hàm tên quen thuộc (`get_prices`, `load_prices`, `get_provider`...), `upgrade_api.load_prices` sẽ **tự động mượn** để dùng chung nguồn dữ liệu. Nếu không tìm thấy, nó lui về dữ liệu mô phỏng nên demo không bao giờ chết vì mất mạng.

---

## Cài đặt: chép file

```text
portfolio/
├── core/
│   ├── upgrade_api.py      ← MỚI  (lớp nối + tiện ích + dữ liệu mô phỏng)
│   ├── alloc_lab.py        ← MỚI  (6 thuật toán phân bổ + co ngót + Black-Litterman)
│   ├── wf_backtest.py      ← MỚI  (walk-forward + chi phí giao dịch)
│   ├── validation.py       ← MỚI  (PSR, Deflated Sharpe, bootstrap)
│   ├── hmm_regime.py       ← MỚI  (Markov ẩn nhận diện chế độ)
│   ├── stress.py           ← MỚI  (kịch bản, stress test, Monte Carlo)
│   └── ...                   (file cũ của bạn giữ nguyên)
├── tabs/
│   ├── tab_lab.py          ← MỚI  (giao diện 5 tab con)
│   └── ...                   (file cũ của bạn giữ nguyên)
├── tests/
│   └── test_upgrade_pack.py ← MỚI (30 test độc lập)
└── app_demo.py             ← MỚI (chạy thử riêng, không đụng app.py)
```

Không cần thêm thư viện mới: gói chỉ dùng `numpy`, `pandas`, `scipy`, `streamlit`, `plotly` — đúng những thứ đã có trong `requirements.txt` của bạn. Mô hình HMM được viết thuần NumPy nên **không cần `hmmlearn`**.

## Chạy thử trước khi tích hợp

```bash
streamlit run app_demo.py       # chạy riêng, app.py của bạn không bị ảnh hưởng
pytest tests/test_upgrade_pack.py -v
```

## Tích hợp vào `app.py` (3 dòng)

```python
from tabs import tab_lab                         # (1) thêm import

tabs = st.tabs([..., "Quant Lab"])               # (2) thêm một tab vào danh sách

with tabs[-1]:                                   # (3) gọi render
    tab_lab.render()
```

Muốn dùng chung dữ liệu đã tải sẵn trong `app.py`:

```python
with tabs[-1]:
    tab_lab.render(prices=prices_df)   # prices_df: index là ngày, cột là mã
```

---

## Gói này sửa chính xác những gì

Bài báo gốc tự liệt kê các điểm yếu của mình. Dưới đây là ánh xạ một–một.

| Điểm yếu bài gốc thừa nhận | Gói nâng cấp xử lý thế nào | File |
|---|---|---|
| Kỳ vọng lợi suất là trung bình lịch sử, rất nhiễu | Co ngót James–Stein + Black–Litterman | `alloc_lab.py` |
| Tối ưu Sharpe bất ổn, cho ra trọng số vô lý | Co ngót Ledoit–Wolf + HRP + bài kiểm định độ ổn định đo bằng số | `alloc_lab.py` |
| Mô hình chế độ là quy tắc tự đặt, cảm tính | HMM Gauss khớp bằng EM, có xác suất và ma trận chuyển trạng thái | `hmm_regime.py` |
| Bỏ qua hoàn toàn chi phí, trượt giá, vòng quay | Tính phí mỗi lần tái cân bằng, trọng số trôi theo giá, phân tích độ nhạy phí, mức phí hoà vốn | `wf_backtest.py` |
| CAPM và phân tích nhân tố quá sơ sài | Hồi quy bội có thống kê t, phân rã lợi nhuận theo nhân tố | `stress.py` |
| Backtest tối ưu và đánh giá trên cùng dữ liệu | Walk-forward có khoảng cấm, bảng so sánh in-sample vs out-of-sample | `wf_backtest.py` |
| Gọi Yahoo Finance lặp đi lặp lại | Cache của Streamlit + nguồn mô phỏng dự phòng | `upgrade_api.py` |

Ngoài ra còn hai thứ bài gốc **không hề nhắc tới** nhưng mọi nhà quản lý quỹ đều đòi:

- **Deflated Sharpe Ratio** — phạt Sharpe theo số cấu hình đã thử.
- **Stress test theo kịch bản** — "nếu SPY giảm 20% thì danh mục mất bao nhiêu?"

---

## Năm tab con và ý nghĩa

### 1. Phân bổ vốn
Sáu thuật toán chạy song song: Chia đều, Phương sai tối thiểu, Max Sharpe, Cân bằng rủi ro, HRP, Đa dạng hoá tối đa. Cột **Số vị thế hiệu dụng** phơi bày ngay danh mục nào chỉ "giả vờ" đa dạng.

Nút *Kiểm định ổn định* tối ưu lại 24 lần trên các đoạn lịch sử con. Trên dữ liệu thử nghiệm, độ lệch chuẩn trọng số của Max Sharpe cao **gấp khoảng 20 lần** HRP và Phương sai tối thiểu — bằng chứng định lượng chứ không phải lời nhận xét suông.

### 2. Walk-forward & chi phí
Kết quả chạy thử trên dữ liệu mô phỏng 2018–2026:

| Phương pháp | Sharpe sau phí | Vòng quay/năm | Phí bào mòn/năm |
|---|---|---|---|
| Chia đều | 1,17 | 0,24 | 0,0% |
| HRP | 0,80 | 0,46 | 0,1% |
| Max Sharpe | 0,61 | **2,52** | **0,5%** |
| Phương sai tối thiểu | 0,58 | 0,36 | 0,1% |

Bảng **in-sample vs out-of-sample** là slide đắt giá nhất khi bảo vệ: Max Sharpe đạt 1,25 khi tối ưu và đánh giá trên cùng dữ liệu, nhưng chỉ còn **0,61** khi làm đúng quy trình — mất gần **52%**.

### 3. Kiểm định
Khai báo trung thực số cấu hình đã thử, mô hình sẽ phạt tương ứng. Trong lần chạy thử: PSR = 0,99 trông rất đẹp, nhưng DSR với 20 lần thử chỉ còn **0,54** — dưới ngưỡng 0,95, nghĩa là chưa đủ bằng chứng để khẳng định có alpha thật. Đây chính là loại kết luận khiêm tốn mà hội đồng đánh giá cao.

Bootstrap theo khối cho khoảng tin cậy giữ được tự tương quan, để báo cáo kiểu "Sharpe 0,80, KTC 95% [0,01; 1,50]".

### 4. Chế độ thị trường
HMM ba trạng thái cho ra ma trận chuyển trạng thái với đường chéo 0,94–0,98 — chế độ dai dẳng đúng như thị trường thật, không nhảy loạn mỗi ngày. Kèm theo số phiên kỳ vọng của mỗi chế độ và tỷ trọng thời gian dài hạn.

Backtest chuyển chế độ **có độ trễ một phiên**, và có test riêng đảm bảo điều đó (`test_regime_backtest_uses_lag`).

### 5. Kịch bản & sức chịu đựng
Ba thanh trượt cho cú sốc SPY / VIX / QQQ, ra ngay mức tổn thất ước tính kèm phân rã theo nhân tố. Thêm phần phát lại các giai đoạn khủng hoảng lịch sử và mô phỏng Monte Carlo dùng phân phối Student-t — vì giả định phân phối chuẩn đánh giá thấp nghiêm trọng xác suất của những phiên sụp đổ.

---

## Phân công cho nhóm bốn người

| Người | Sở hữu | Nội dung trình bày |
|---|---|---|
| 1 | `upgrade_api.py`, cache, CI | Kiến trúc dữ liệu nhiều nguồn, chế độ ngoại tuyến, khả năng tái lập |
| 2 | `alloc_lab.py` | Vì sao Max Sharpe bất ổn và co ngót/HRP chữa ra sao |
| 3 | `wf_backtest.py`, `validation.py` | Rò rỉ dữ liệu tương lai, chi phí giao dịch, kỹ năng hay may mắn |
| 4 | `hmm_regime.py`, `stress.py`, `tab_lab.py` | Mô hình chế độ xác suất và phòng kịch bản |

Mỗi người sở hữu file riêng, ghép với nhau qua `pd.DataFrame` nên không ai chặn ai.

## Câu hỏi hội đồng hay hỏi và cách trả lời

**"Sao Sharpe thấp hơn bài gốc?"**
Vì bài gốc tối ưu và đánh giá trên cùng một tập dữ liệu, lại không trừ chi phí. Bảng in-sample vs out-of-sample trong tab 2 định lượng chính xác khoảng chênh đó.

**"Chia đều lại thắng à?"**
Đúng, và đó là kết quả đã được ghi nhận rộng rãi trong tài liệu học thuật khi ước lượng tham số quá nhiễu. Nhóm giữ nguyên kết quả này thay vì tinh chỉnh cho đẹp.

**"Vì sao không dùng `hmmlearn`?"**
Để tránh thêm phụ thuộc nặng và để nhóm chứng minh hiểu thuật toán EM. Mã nguồn có đầy đủ forward–backward, Baum–Welch và Viterbi, đã kiểm thử tính dai dẳng của chế độ.

**"Dữ liệu mô phỏng có làm kết quả vô nghĩa không?"**
Dữ liệu mô phỏng chỉ là phương án dự phòng khi không có mạng. Nó được dựng theo mô hình nhân tố có chuyển chế độ Markov nên giữ được tương quan và drawdown thật, đủ để kiểm thử. Khi có mạng, gói tự dùng nguồn dữ liệu thật của dự án.
