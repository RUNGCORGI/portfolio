# Phân công & bằng chứng đóng góp cá nhân

Giảng viên chấm điểm cá nhân. Mỗi thành viên có: branch riêng, file sở hữu riêng,
file test riêng, tab demo riêng, và phần phát triển thêm riêng.

Kiểm chứng nhanh:

```bash
git log --author="<tên>" --stat          # ai viết file nào, bao nhiêu dòng
git shortlog -sne                         # tổng số commit từng người
cat .github/CODEOWNERS                    # bản đồ quyền sở hữu file
docker compose run --rm --profile ci tests -k p3   # chạy riêng test của P3
```

| | Họ tên | Branch | File sở hữu | Test | Tab demo |
|---|---|---|---|---|---|
| P1 | _(điền)_ | `feat/infra-p1` | `core/data.py`, `core/cache.py`, `core/fixtures.py`, `Dockerfile`, `docker-compose.yml` | `tests/test_p1_data.py` (7) | Panel "Tầng dữ liệu" |
| P2 | _(điền)_ | `feat/portfolio-p2` | `core/portfolio.py`, `core/costs.py`, `core/sizing.py`, `tabs/tab_portfolio.py`, `tabs/tab_signals.py` | `tests/test_p2_portfolio.py` (19) | Portfolio, Signals |
| P3 | _(điền)_ | `feat/risk-p3` | `core/risk.py`, `core/factors.py`, `tabs/tab_risk.py`, `tabs/tab_factors.py` | `tests/test_p3_risk.py` (11) | Risk, Factors |
| P4 | _(điền)_ | `feat/optim-p4` | `core/optimizer.py`, `core/regime.py`, `tabs/tab_optimize.py`, `tabs/tab_backtest.py` | `tests/test_p4_optimizer.py` (13) | Optimize, Backtest |
| P5 / AI | _(điền)_ | `feat/quantamental` | `quantamental/*`, `tabs/tab_quantamental.py` | `tests/test_quantamental.py` (15) | Quantamental AI (Tab 7) |

File dùng chung: `core/contracts.py` (khóa, cần cả 4 duyệt), `app.py` (sửa tối thiểu).

## Quy trình làm việc

1. **Ngày 1** — cả nhóm chốt `core/contracts.py`, P1 commit `data/sample_*.csv`.
   Từ lúc này 3 người còn lại code được ngay, không chờ ai, không cần mạng.
2. **Ngày 2–5** — 4 branch chạy song song. Chỉ sửa file mình sở hữu.
3. **Ngày 6** — merge PR, chạy `pytest` tích hợp, tập chuyển tiếp giữa các tab.

## Quy tắc PR

- Một PR = một branch = một người.
- PR chạm `core/contracts.py` cần approve của cả 4.
- CI phải xanh (46 test) trước khi merge.
