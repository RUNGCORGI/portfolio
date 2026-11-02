# Bloomberg-Style Portfolio Terminal — Nhóm 4

Refactor bài báo *"Building an Awesome Portfolio Management App in Streamlit from Scratch"* (Alexzap, 06/2026) từ một file `myapp13.py` 800 dòng thành kiến trúc module có test, chạy bằng một lệnh Docker.

```bash
docker compose up          # -> http://localhost:8501
docker compose run --rm --profile ci tests    # 46 test của cả 4 thành viên
```

Không có Docker:

```bash
pip install -r requirements.txt
python -m core.fixtures     # sinh dữ liệu mẫu offline
streamlit run app.py
```

---

## Vì sao phải refactor

Bản in `myapp13.py` trong bài báo **không chạy được**: các tab gọi tới `corr`, `rolling_vol`, `drawdown`, `rolling_beta`, `beta_latest`, `signal_df`, `final_signal`, `portfolio_total_return`, `alpha`, `regime_state` — nhưng những khối định nghĩa chúng đã bị cắt mất khi tác giả copy-paste từ `myapp10.py` sang. Tab 5 còn dùng `col4` trong khi chỉ khai báo `st.columns(3)`.

Nhóm bắt đầu từ đó: dựng lại cho chạy, tách thành module theo trách nhiệm, rồi mỗi thành viên phát triển thêm một lớp mà bài gốc còn thiếu.

## Kiến trúc

```
app.py                  Điều phối UI (cả nhóm)
core/
  contracts.py          ★ FILE KHÓA — hợp đồng dữ liệu giữa 4 người
  data.py  cache.py  fixtures.py        P1
  portfolio.py  costs.py  sizing.py     P2
  risk.py  factors.py                   P3
  optimizer.py  regime.py               P4
quantamental/           ★ Module Quantamental AI & Dual-RAG
tabs/                   Mỗi tab do một người sở hữu (bao gồm tab_quantamental)
tests/                  Kiểm thử tự động toàn diện
```

Luồng phụ thuộc duy nhất: **P1 → P2 → (P3 ∥ P4)**. P3 và P4 chạy song song, không đụng nhau. Mọi giao tiếp đi qua `core/contracts.py`; được thêm field mới, cấm đổi/xóa field cũ.

## Phân công và phần phát triển thêm

| | Phụ trách | Module sở hữu | Phát triển thêm (không có trong bài gốc) |
|---|---|---|---|
| **P1** | Dữ liệu & hạ tầng | `data.py`, `cache.py`, `fixtures.py`, `Dockerfile` | **(1)** Cache 2 tầng (RAM + đĩa) thay cho 4–5 lần gọi `yf.download` trùng lặp, có fallback offline · **(2)** Docker multi-stage + non-root user + healthcheck |
| **P2** | Portfolio engine | `portfolio.py`, `costs.py`, `sizing.py` | **(1)** Chi phí giao dịch theo turnover + tái cân bằng định kỳ (bài gốc bỏ qua hoàn toàn) · **(2)** Biến tỷ trọng % thành phiếu lệnh thực thi: số cổ phiếu nguyên lô, tiền mặt dư, sai lệch do làm tròn, xuất CSV |
| **P3** | Rủi ro & nhân tố | `risk.py`, `factors.py` | **(1)** 8 chỉ số đuôi rủi ro (VaR/CVaR/Sortino/Calmar/DD duration/skew/kurtosis) · **(2)** Hồi quy đa nhân tố OLS có t-stat và R² thay cho tương quan trượt |
| **P4** | Tối ưu & regime | `optimizer.py`, `regime.py` | **(1)** Thêm Min-Variance, Risk Parity, HRP so sánh với Max Sharpe · **(2)** Walk-forward backtest bóc tách look-ahead bias |
| **Quantamental** | Quantamental AI & Dual-RAG | `quantamental/`, `tabs/tab_quantamental.py` | **(1)** Phân tích đa chiều: Tín hiệu kỹ thuật (T) + RAG Cảm xúc báo chí (S) + RAG Nội tại BCTC (F) · **(2)** Ma trận chân lý 8 trạng thái loại bỏ Bull Trap · **(3)** Thực nghiệm đối chứng A/B Testing |

## Ba lỗi phương pháp của bài gốc đã sửa

1. **Look-ahead bias.** Bài gốc tối ưu và đo hiệu suất trên cùng một tập dữ liệu. `core/regime.walk_forward()` tối ưu trên 252 phiên rồi áp dụng cho 63 phiên kế tiếp — chênh lệch Sharpe in-sample vs out-of-sample hiện ngay trên tab Backtest.
2. **Regime tính trên chuỗi sai.** `myapp10.py` so sánh MA50 vs MA200 của chuỗi **lợi suất** SPY. Tín hiệu xu hướng phải tính trên chuỗi **giá**.
3. **Tín hiệu regime không trễ.** Chiến lược chuyển trạng thái dùng `state.shift(1)` — quyết định hôm nay chỉ được dùng thông tin đến hết hôm qua.

## Chế độ trình bày

Sidebar có selector `Chế độ trình bày`. Khi tới lượt ai, chọn chế độ của người đó, app chỉ hiện tab tương ứng. Một sản phẩm thống nhất, nhưng demo cá nhân vẫn sạch.

## Demo không cần mạng

Bật toggle **"Dùng dữ liệu mẫu (offline)"**. `core/fixtures.py` sinh dữ liệu tổng hợp bằng GBM có cấu trúc giống thật (tech beta cao, VIX nghịch biến SPY), seed cố định nên tái lập 100%.

## Nguồn

Alexzap — *Building an Awesome Portfolio Management App in Streamlit from Scratch* (06/2026). Code trong repo này là bản viết lại của nhóm, không sao chép nguyên văn.
