# Kịch bản demo 25 phút

Chuẩn bị: `docker compose up -d` TRƯỚC khi vào phòng. Bật sẵn toggle
"Dùng dữ liệu mẫu (offline)" để không phụ thuộc wifi.

| Phút | Người | Việc làm trên máy | Câu chốt |
|---|---|---|---|
| 0:00–1:00 | P1 | Mở slide kiến trúc | "Bài báo dừng ở file 800 dòng và bản in đó thiếu 4 khối nên không chạy. Nhóm em bắt đầu từ đó." |
| 1:00–6:00 | P1 | `docker compose up` live → chọn chế độ **P1** → chỉ "Nguồn: Cache (0 request mạng)", đo ms. Chạy `make size` | "Docker không làm app nhanh hơn, nó làm app tái lập được." |
| 6:00–11:30 | P2 | Chế độ **P2** → kéo slider weight → Alpha đổi. Bật phí 30bps + rebalance Tháng → đường "trước phí" tách ra. Mở phiếu lệnh: đổi US → HOSE, vốn 20.000$ → chỉ sai lệch 33% | "Bài gốc báo lãi 576% mà chưa trừ đồng phí nào — và dừng ở tỷ trọng %, chưa ai mua được 33,3% NVDA." |
| 11:30–17:30 | P3 | Chế độ **P3** → bảng 8 chỉ số rủi ro → tab Factors, bảng OLS, chỉ cột t-stat | "Phơi nhiễm chỉ có ý nghĩa khi |t| > 2 — bài gốc không kiểm định gì cả." |
| 17:30–23:30 | P4 | Chế độ **P4** → bảng so sánh 5 phương pháp (chỉ cột Max w%) → walk-forward, chỉ delta Sharpe | "Sharpe 1.87 của bài gốc là in-sample. Out-of-sample thấp hơn rõ rệt." |
| 23:30–25:00 | Cả nhóm | `docker compose run --rm --profile ci tests` → 46 test xanh | "Toàn bộ chạy trong đúng môi trường của app." |

## Phương án dự phòng

- Mất mạng → đã dùng dữ liệu mẫu, không ảnh hưởng.
- Docker lỗi trên máy phòng học → `streamlit run app.py` trực tiếp.
- Máy chiếu hỏng → mỗi người có video demo 60 giây quay sẵn.

## Câu hỏi nên tự chuẩn bị

- **P1**: Tại sao Docker chứ không gửi file .py? Cache hết hạn thì sao? Multi-stage tiết kiệm được gì?
- **P2**: Equal-weight 3 mã tech có thực sự là đa dạng hóa? 10bps lấy ở đâu ra? Rebalance nhiều có tốt hơn không?
- **P3**: VaR lịch sử và tham số khác nhau chỗ nào? Beta 20 ngày có quá nhiễu? Vì sao beta VIX âm?
- **P4**: Vì sao Max Sharpe hay dồn vào 1–2 mã? HRP khác Risk Parity ở đâu? Walk-forward chọn 252/63 dựa trên gì?
