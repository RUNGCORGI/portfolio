"""
quantamental/truth_table.py — Bộ phân loại Ma trận Chân lý 8 trạng thái [T, S, F].
Kết hợp Tín hiệu Kỹ thuật (T), Cảm xúc Báo chí (S) và Nội tại BCTC (F).
"""
from __future__ import annotations

from typing import Dict, Tuple
from quantamental.models import FundamentalResult, QuantamentalSignal, SentimentResult

# Bảng định nghĩa 8 trạng thái [T, S, F]
TRUTH_TABLE_RULES: Dict[Tuple[int, int, int], Dict[str, str]] = {
    (1, 0, 0): {
        "name": "Bẫy tăng giá (Bull Trap / Kéo xả kỹ thuật)",
        "color": "#EF553B",  # Đỏ cảnh báo
        "desc": "Giá vượt MA tạo tín hiệu mua giả nhưng truyền thông im ắng và BCTC rỗng ruột. Dấu hiệu điển hình của việc tạo sóng ảo để thoát hàng của đội lái.",
        "recommendation": "🚨 CẢNH BÁO NGUY HIỂM / LOẠI BỎ KHỎI DANH MỤC",
        "action": "Ép tỷ trọng w = 0% (Blacklist)"
    },
    (1, 1, 1): {
        "name": "Tín Hiệu Vàng (Quality Momentum / Siêu cổ phiếu)",
        "color": "#00CC96",  # Xanh lục rực rỡ
        "desc": "Đồng thuận tuyệt đối: Doanh nghiệp tốt thật, truyền thông lan tỏa tích cực và dòng tiền kỹ thuật đẩy giá mạnh mẽ.",
        "recommendation": "⭐ GIẢI NGÂN MẠNH VỚI TỶ TRỌNG CAO NHẤT",
        "action": "Ưu tiên phân bổ tối đa (Max Sharpe)"
    },
    (1, 1, 0): {
        "name": "Hùa theo đám đông (FOMO / Đầu cơ bong bóng)",
        "color": "#FFA15A",  # Cam cảnh báo
        "desc": "Giá tăng mạnh và truyền thông thổi phồng rầm rộ, nhưng BCTC không có nền tảng lợi nhuận thật (sóng tin đồn, cổ phiếu meme).",
        "recommendation": "⚠️ CHỈ LƯỚT SÓNG NGẮN HẠN CÓ STOP-LOSS CHẶT",
        "action": "Giới hạn tỷ trọng < 10%, sẵn sàng chốt lời"
    },
    (1, 0, 1): {
        "name": "Ngọc Trong Đá (Hidden Gem / Smart Money gom)",
        "color": "#19D3F3",  # Xanh ngọc / Cyan
        "desc": "Doanh nghiệp nội tại rất tốt và giá chớm bứt phá, nhưng truyền thông chưa chú ý, đám đông còn thờ ơ. Vùng gom lý tưởng của dòng tiền thông minh.",
        "recommendation": "💎 VÙNG MUA GOM TÍCH LŨY AN TOÀN NHẤT",
        "action": "Gia tăng tỷ trọng bền vững"
    },
    (0, 1, 1): {
        "name": "Vùng Nén Tích Lũy (Spring Coil / Chờ bứt phá)",
        "color": "#AB63FA",  # Tím tích lũy
        "desc": "Cơ bản doanh nghiệp rất tốt, truyền thông đánh giá cao nhưng giá đang đi ngang/điều chỉnh kỹ thuật dưới đường MA.",
        "recommendation": "⏳ ĐƯA VÀO DANH SÁCH THEO DÕI ĐẶC BIỆT (WATCHLIST)",
        "action": "Giữ tỷ trọng thăm dò, chờ T=1 để mua thêm"
    },
    (0, 1, 0): {
        "name": "Bánh Vẽ Truyền Thông (Media Hype / Lùa gà)",
        "color": "#E377C2",  # Hồng tím
        "desc": "Truyền thông PR rầm rộ nhưng giá không tăng và doanh nghiệp nội tại rỗng ruột. Chiêu bài truyền thông tạo thanh khoản ảo.",
        "recommendation": "📢 NÉ TRÁNH CHIÊU BÀI PR TRUYỀN THÔNG",
        "action": "Không tham gia"
    },
    (0, 0, 1): {
        "name": "Bẫy Giá Trị (Value Trap / Rơi tự do)",
        "color": "#B6E880",  # Xanh lá nhạt
        "desc": "Doanh nghiệp cơ bản tốt nhưng giá đang cắm đầu rơi và tin tức im ắng. Chưa có dòng tiền lớn bảo trợ xu hướng.",
        "recommendation": "🪤 KHÔNG BẮT DAO RƠI, ĐỢI TẠO ĐÁY KỸ THUẬT",
        "action": "Quan sát, chờ tín hiệu kỹ thuật đảo chiều"
    },
    (0, 0, 0): {
        "name": "Né Tránh Hoàn Toàn (Strong Avoid)",
        "color": "#7F7F7F",  # Xám
        "desc": "Cả 3 trục đều tiêu cực: Kỹ thuật gãy xu hướng, truyền thông ảm đạm và BCTC thua lỗ.",
        "recommendation": "⛔ ĐỨNG NGOÀI HOÀN TOÀN / BÁN SẠCH NẾU CÒN GIỮ",
        "action": "Tỷ trọng w = 0%"
    },
}


def classify_quantamental_signal(
    ticker: str,
    technical_flag: int,
    sentiment: SentimentResult,
    fundamental: FundamentalResult
) -> QuantamentalSignal:
    """
    Ánh xạ bộ ba [T, S, F] vào Ma trận Chân lý để tạo tín hiệu Quantamental hoàn chỉnh.
    """
    T = int(technical_flag)
    S = int(sentiment.flag)
    F = int(fundamental.flag)

    pattern_tuple = (T, S, F)
    rule = TRUTH_TABLE_RULES.get(pattern_tuple, {
        "name": "Không xác định",
        "color": "#7F7F7F",
        "desc": "Chưa có quy tắc phù hợp",
        "recommendation": "Quan sát",
        "action": "Giữ nguyên"
    })

    return QuantamentalSignal(
        ticker=ticker,
        T=T,
        S=S,
        F=F,
        pattern_code=f"[{T}, {S}, {F}]",
        pattern_name=rule["name"],
        badge_color=rule["color"],
        sentiment=sentiment,
        fundamental=fundamental,
        description=rule["desc"],
        recommendation=rule["recommendation"],
        weight_action=rule["action"]
    )

