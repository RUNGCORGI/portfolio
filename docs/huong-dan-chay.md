# Hướng dẫn tải và chạy trên máy tính

Có hai cách. **Cách A (Docker)** là cách dùng khi demo trước lớp. **Cách B (Python)** là cách nhẹ hơn, dùng khi code hằng ngày hoặc khi máy phòng học không cài được Docker.

---

## Bước 0 — Tải và giải nén

1. Bấm nút tải `portfolio-terminal.zip` trong khung chat.
2. Giải nén.
   - **Windows:** chuột phải → *Extract All…* → chọn thư mục dễ nhớ, ví dụ `C:\Users\<tên>\Desktop`
   - **macOS:** nháy đúp file zip
3. Sau khi giải nén sẽ có thư mục `portfolio-terminal`. Mở terminal **ngay trong thư mục này**:
   - **Windows:** vào thư mục, gõ `cmd` vào thanh địa chỉ của File Explorer rồi Enter
   - **macOS:** chuột phải thư mục → *New Terminal at Folder*

Kiểm tra đúng chỗ:

```bash
dir        # Windows
ls         # macOS/Linux
```

Phải thấy `app.py`, `Dockerfile`, `core`, `tabs`, `tests`.

---

## Cách A — Chạy bằng Docker (khuyến nghị cho buổi demo)

### A1. Cài Docker Desktop

Tải tại [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/), cài rồi **mở ứng dụng Docker Desktop lên**. Chờ biểu tượng cá voi ở thanh trạng thái chuyển sang xanh (đang chạy).

> Windows: nếu báo lỗi WSL 2, mở PowerShell quyền Admin, chạy `wsl --install`, khởi động lại máy.

### A2. Chạy

```bash
docker compose up --build
```

Lần đầu mất khoảng 2–3 phút (tải Python base image + cài thư viện). Các lần sau vài giây.

Khi thấy dòng:

```
terminal  |   You can now view your Streamlit app in your browser.
terminal  |   URL: http://0.0.0.0:8501
```

→ mở trình duyệt vào **http://localhost:8501**

### A3. Chạy test (slide ghi điểm của nhóm)

Mở terminal **thứ hai** trong cùng thư mục:

```bash
docker compose run --rm --profile ci tests
```

Kết quả mong đợi: `54 passed`.

### A4. Dừng

Nhấn `Ctrl + C` ở terminal đang chạy, rồi:

```bash
docker compose down
```

### Các lệnh tắt khác (macOS/Linux)

```bash
make up       # chạy nền
make test     # chạy test
make size     # xem kích thước image — P1 dùng để demo
make logs     # xem log
make down     # dừng
```

---

## Cách B — Chạy trực tiếp bằng Python

### B1. Cài Python 3.11

Tải tại [python.org/downloads](https://www.python.org/downloads/).

> **Windows: nhớ tick ô "Add Python to PATH"** ở màn hình cài đầu tiên. Quên bước này là nguyên nhân lỗi phổ biến nhất.

Kiểm tra:

```bash
python --version        # Windows
python3 --version       # macOS/Linux
```

### B2. Tạo môi trường ảo

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Thành công khi đầu dòng lệnh có chữ `(.venv)`.

### B3. Cài thư viện

```bash
pip install -r requirements.txt
```

Mất 1–3 phút.

### B4. Sinh dữ liệu mẫu offline

```bash
python -m core.fixtures
```

### B5. Chạy app

```bash
streamlit run app.py
```

Trình duyệt tự mở tại http://localhost:8501. Nếu không, mở tay.

### B6. Chạy test

```bash
python -m pytest -v
```

### B7. Thoát

`Ctrl + C` để dừng app, rồi `deactivate` để thoát môi trường ảo.

Những lần sau chỉ cần: mở terminal trong thư mục → kích hoạt venv (B2, dòng thứ hai) → `streamlit run app.py`.

---

## Dùng app lần đầu

Sidebar bên trái có **"Chế độ trình bày"** — chọn `P1`, `P2`, `P3`, `P4` để xem phần của từng người, hoặc `Toàn bộ` để thấy cả 6 tab.

Toggle **"Dùng dữ liệu mẫu (offline)"** đang BẬT sẵn. Giữ nguyên khi demo để không phụ thuộc wifi. Tắt nếu muốn lấy dữ liệu thật từ Yahoo Finance (cần mạng, lần đầu chậm hơn).

Thử nhanh:
- Kéo slider tỷ trọng → các chỉ số đổi ngay
- Chỉnh "Phí mỗi chiều (bps)" lên 30 → tab Portfolio xuất hiện đường "trước phí"
- Tab Optimize → đổi phương pháp sang `hrp` → so bảng 5 phương pháp
- Tab Backtest → mở mục walk-forward → xem chênh lệch Sharpe

---

## Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `docker: command not found` | Chưa cài hoặc chưa mở Docker Desktop | Mở Docker Desktop, chờ biểu tượng xanh |
| `Cannot connect to the Docker daemon` | Docker Desktop chưa chạy | Mở ứng dụng Docker Desktop |
| `port is already allocated` | Cổng 8501 đang bận | `docker compose down`, hoặc sửa `docker-compose.yml` thành `"8502:8501"` |
| `'python' is not recognized` (Windows) | Quên tick Add to PATH | Cài lại Python, tick ô đó |
| `ModuleNotFoundError: streamlit` | Chưa kích hoạt venv | Chạy lại bước B2 dòng kích hoạt |
| `No module named core` | Đang ở sai thư mục | `cd` vào thư mục chứa `app.py` |
| Trang trắng / xoay mãi | App đang tải dữ liệu lần đầu | Chờ 10–20 giây, hoặc bật toggle dữ liệu mẫu |
| `YFRateLimitError` / tải dữ liệu lỗi | Yahoo Finance chặn tạm | Bật toggle **"Dùng dữ liệu mẫu (offline)"** |

Muốn xóa cache làm lại từ đầu:

```bash
rm -rf data/cache        # macOS/Linux
rmdir /s data\cache      # Windows
```

---

## Checklist trước buổi thuyết trình

- [ ] `docker compose up -d` chạy **trước** khi vào phòng (tránh chờ build)
- [ ] Mở sẵn http://localhost:8501, kiểm tra cả 5 chế độ hiện đúng
- [ ] Bật toggle "Dùng dữ liệu mẫu" — không phụ thuộc wifi
- [ ] Mở sẵn terminal thứ hai để chạy `pytest` khi cần
- [ ] Mỗi người quay video demo 60 giây dự phòng
