"""
quantamental/rag_fundamental.py — RAG Đánh giá Sức khỏe Nội tại BCTC & Tiềm lực Doanh nghiệp (Flag F).
Đánh giá chất lượng lợi nhuận, dòng tiền kinh doanh (OCF) và an toàn nợ vay.
"""
from __future__ import annotations

from typing import List
from quantamental.models import FinancialMetrics, FundamentalResult


def analyze_fundamentals(ticker: str, metrics: FinancialMetrics) -> FundamentalResult:
    """
    RAG Agent phân tích bảng số liệu BCTC theo các trụ cột sức khỏe tài chính:
    1. Dòng tiền thuần từ HĐKD (Operating Cash Flow - OCF)
    2. Tăng trưởng doanh thu (Revenue Growth)
    3. Biên lợi nhuận ròng (Net Profit Margin)
    4. An toàn cấu trúc vốn (Nợ vay / Vốn chủ sở hữu)
    """
    passed: List[str] = []
    failed: List[str] = []
    points = 0
    total_criteria = 4

    # 1. Dòng tiền OCF
    if metrics.ocf > 0:
        points += 1
        passed.append(f"Dòng tiền thuần HĐKD (OCF) dương đạt {metrics.ocf:,.1f} {metrics.currency} (Lợi nhuận đi kèm tiền thật)")
    else:
        failed.append("Dòng tiền kinh doanh (OCF) âm — Rủi ro thâm hụt vốn lưu động")

    # 2. Tăng trưởng doanh thu
    if metrics.revenue_growth_yoy > 3.0:
        points += 1
        passed.append(f"Tăng trưởng doanh thu tích cực (+{metrics.revenue_growth_yoy:.1f}% YoY)")
    elif metrics.revenue_growth_yoy >= 0.0:
        points += 0.5
        passed.append(f"Doanh thu duy trì ổn định (+{metrics.revenue_growth_yoy:.1f}% YoY)")
    else:
        failed.append(f"Doanh thu sụt giảm ({metrics.revenue_growth_yoy:.1f}% YoY)")

    # 3. Biên lợi nhuận ròng
    if metrics.net_margin >= 10.0:
        points += 1
        passed.append(f"Biên lợi nhuận ròng cao ({metrics.net_margin:.1f}%) — Thể hiện lợi thế cạnh tranh mạnh")
    elif metrics.net_margin > 4.0:
        points += 0.5
        passed.append(f"Biên lợi nhuận ròng ở mức chấp nhận được ({metrics.net_margin:.1f}%)")
    else:
        failed.append(f"Biên lợi nhuận ròng mỏng ({metrics.net_margin:.1f}%) — Dễ bị tổn thương khi chi phí tăng")

    # 4. An toàn nợ vay
    if metrics.debt_to_equity < 1.2:
        points += 1
        passed.append(f"Cấu trúc vốn an toàn (Nợ/VCSH = {metrics.debt_to_equity:.2f} < 1.2)")
    elif metrics.debt_to_equity < 2.0:
        points += 0.5
        passed.append(f"Đòn bẩy tài chính ở mức trung bình (Nợ/VCSH = {metrics.debt_to_equity:.2f})")
    else:
        failed.append(f"Đòn bẩy tài chính cao (Nợ/VCSH = {metrics.debt_to_equity:.2f}) — Rủi ro lãi vay")

    score = round(points / total_criteria, 2)
    flag_f = 1 if score >= 0.60 else 0

    if score >= 0.75:
        label = "Vững chắc / Xuất sắc"
    elif score >= 0.50:
        label = "Khá / Ổn định"
    else:
        label = "Bất ổn / Rỗng ruột"

    reasoning = (
        f"Doanh nghiệp đạt {points}/{total_criteria} điểm tiêu chuẩn sức khỏe tài chính "
        f"(Điểm F: {score:.2f} -> {'ĐẠT NỘI TẠI F=1' if flag_f == 1 else 'KHÔNG ĐẠT F=0'}). "
        f"Lợi nhuận ròng: {metrics.net_income:,.1f} {metrics.currency}, P/E: {metrics.pe_ratio:.1f}x."
    )

    return FundamentalResult(
        ticker=ticker,
        score=score,
        flag=flag_f,
        label=label,
        metrics=metrics,
        criteria_passed=passed,
        criteria_failed=failed,
        reasoning=reasoning
    )

