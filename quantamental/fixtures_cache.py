"""
quantamental/fixtures_cache.py — Dữ liệu dự phòng offline cho tin tức và BCTC.
Đảm bảo hệ thống luôn hoạt động mượt mà khi mất mạng hoặc không kết nối được API bên ngoài.
"""
from __future__ import annotations

from typing import Dict, List
from quantamental.models import FinancialMetrics, NewsItem

MOCK_NEWS: Dict[str, List[NewsItem]] = {
    "AAPL": [
        NewsItem(
            title="Apple công bố doanh thu dịch vụ và iPhone quý gần nhất đạt kỷ lục mới",
            link="https://finance.yahoo.com/quote/AAPL",
            pub_date="2026-10-25",
            source="Reuters",
            snippet="Doanh thu mảng dịch vụ và hệ sinh thái Apple Intelligence tiếp tục tăng trưởng hai con số."
        ),
        NewsItem(
            title="Các chuyên gia phân tích nâng định giá mục tiêu của Apple lên 370 USD",
            link="https://finance.yahoo.com/quote/AAPL",
            pub_date="2026-10-24",
            source="Bloomberg",
            snippet="J.P. Morgan và Morgan Stanley duy trì xếp hạng Overweight với triển vọng nâng cấp thiết bị mạnh mẽ."
        ),
        NewsItem(
            title="Apple mở rộng đối tác sản xuất chip và chuỗi cung ứng tại Đông Nam Á",
            link="https://finance.yahoo.com/quote/AAPL",
            pub_date="2026-10-20",
            source="Wall Street Journal",
            snippet="Chiến lược đa dạng hóa chuỗi cung ứng giúp Apple tối ưu hóa chi phí và giảm thiểu rủi ro."
        ),
    ],
    "MSFT": [
        NewsItem(
            title="Microsoft Azure ghi nhận tăng trưởng vượt bậc nhờ nhu cầu AI doanh nghiệp",
            link="https://finance.yahoo.com/quote/MSFT",
            pub_date="2026-10-26",
            source="CNBC",
            snippet="Dịch vụ đám mây Azure tiếp tục là động lực tăng trưởng chính với biên lợi nhuận hoạt động ổn định."
        ),
        NewsItem(
            title="Microsoft mở rộng thỏa thuận hợp tác bản quyền và tích hợp Copilot",
            link="https://finance.yahoo.com/quote/MSFT",
            pub_date="2026-10-23",
            source="TechCrunch",
            snippet="Hàng triệu người dùng doanh nghiệp đang áp dụng gói bản quyền Microsoft 365 Copilot."
        ),
    ],
    "NVDA": [
        NewsItem(
            title="Nvidia tiếp tục bùng nổ nhu cầu chip GPU thế hệ mới từ các trung tâm dữ liệu",
            link="https://finance.yahoo.com/quote/NVDA",
            pub_date="2026-10-27",
            source="Financial Times",
            snippet="Đơn đặt hàng chip AI Blackwell đã kín chỗ trong các quý tới từ các tập đoàn công nghệ lớn."
        ),
        NewsItem(
            title="Phố Wall sôi động trước dự báo tăng trưởng doanh thu quý 3 của Nvidia",
            link="https://finance.yahoo.com/quote/NVDA",
            pub_date="2026-10-24",
            source="MarketWatch",
            snippet="Tâm lý thị trường hưng phấn cao độ, định giá P/E đang ở mức cao nhưng được hỗ trợ bởi tăng trưởng thực."
        ),
    ],
    "DGC": [
        NewsItem(
            title="DGC: Dự án Nghi Sơn mở rộng và giá phốt pho vàng duy trì mức cao",
            link="https://cafef.vn",
            pub_date="2026-10-25",
            source="CafeF",
            snippet="Hóa chất Đức Giang tiếp tục duy trì lượng tiền mặt ròng dồi dào, biên lợi nhuận gộp trên 35%."
        ),
        NewsItem(
            title="Hóa chất Đức Giang đẩy mạnh xuất khẩu nguyên liệu cho chuỗi bán dẫn toàn cầu",
            link="https://vietstock.vn",
            pub_date="2026-10-20",
            source="Vietstock",
            snippet="Sản phẩm phốt pho vàng điện tử tinh khiết của DGC được các nhà sản xuất chip quốc tế đánh giá cao."
        ),
    ],
    "VNM": [
        NewsItem(
            title="Vinamilk đẩy mạnh mở rộng thị phần xuất khẩu sang khu vực Trung Đông",
            link="https://cafef.vn",
            pub_date="2026-10-22",
            source="VnExpress",
            snippet="Dòng tiền từ hoạt động kinh doanh ổn định, tỷ lệ chi trả cổ tức bằng tiền mặt trên 70%."
        ),
        NewsItem(
            title="Thị trường sữa trong nước duy trì ổn định, giá nguyên liệu thức ăn chăn nuôi hạ nhiệt",
            link="https://vietstock.vn",
            pub_date="2026-10-18",
            source="Báo Đầu Tư",
            snippet="Vinamilk tiếp tục tối ưu hóa chi phí vận hành và quản trị biên lợi nhuận tốt."
        ),
    ],
    "VCB": [
        NewsItem(
            title="Vietcombank tiếp tục dẫn đầu toàn ngành về quy mô lợi nhuận và chất lượng tài sản",
            link="https://cafef.vn",
            pub_date="2026-10-26",
            source="CafeF",
            snippet="Tỷ lệ bao phủ nợ xấu vượt 200%, CASA ổn định ở mức cao giúp tối ưu hóa chi phí vốn."
        ),
        NewsItem(
            title="VCB hoàn tất kế hoạch tăng vốn điều lệ và chi trả cổ tức",
            link="https://vietstock.vn",
            pub_date="2026-10-22",
            source="Vietstock",
            snippet="Ngân hàng củng cố hệ số an toàn vốn CAR theo chuẩn Basel III, khẳng định vị thế ngân hàng hàng đầu."
        ),
    ],
    "FPT": [
        NewsItem(
            title="FPT ghi nhận tăng trưởng mạnh mẽ mảng xuất khẩu phần mềm và chuyển đổi số AI",
            link="https://cafef.vn",
            pub_date="2026-10-25",
            source="VnExpress",
            snippet="Doanh thu ký mới tại thị trường Nhật Bản và Bắc Mỹ tăng trưởng hơn 25%, liên minh AI mở rộng."
        ),
        NewsItem(
            title="FPT mở rộng hệ sinh thái đào tạo công nghệ bán dẫn và hợp tác quốc tế",
            link="https://vietstock.vn",
            pub_date="2026-10-20",
            source="Báo Đầu Tư",
            snippet="Chiến lược chuyển dịch lên chuỗi giá trị cao giúp FPT duy trì biên lợi nhuận vững chắc."
        ),
    ],
    "HPG": [
        NewsItem(
            title="Hòa Phát vận hành giai đoạn mới dự án Dung Quất, sản lượng thép cuộn cán nóng HRC tăng vọt",
            link="https://cafef.vn",
            pub_date="2026-10-24",
            source="CafeF",
            snippet="Hiệu suất vận hành lò cao tối ưu giúp biên lợi nhuận cải thiện rõ rệt, dòng tiền kinh doanh dương lớn."
        ),
        NewsItem(
            title="Thị trường xây dựng phục hồi, sức tiêu thụ thép nội địa và xuất khẩu của HPG tăng trưởng",
            link="https://vietstock.vn",
            pub_date="2026-10-19",
            source="Vietstock",
            snippet="Nhu cầu đầu tư công và bất động sản công nghiệp hỗ trợ vững chắc cho sản lượng tiêu thụ thép."
        ),
    ],
    "VIC": [
        NewsItem(
            title="Tập đoàn Vingroup công bố doanh thu và lợi nhuận hợp nhất tăng trưởng mạnh mẽ",
            link="https://cafef.vn",
            pub_date="2026-10-26",
            source="CafeF",
            snippet="Doanh số bàn giao bất động sản Vinhomes và sản lượng bàn giao xe điện VinFast bứt phá kỷ lục mới."
        ),
        NewsItem(
            title="VinFast mở rộng mạng lưới trạm sạc và nhà máy tại các thị trường quốc tế trọng điểm",
            link="https://vietstock.vn",
            pub_date="2026-10-22",
            source="Vietstock",
            snippet="Dòng tiền từ hoạt động kinh doanh cải thiện rõ rệt, hệ sinh thái công nghệ - dịch vụ vận hành hiệu quả."
        ),
    ],
}

MOCK_FINANCIALS: Dict[str, FinancialMetrics] = {
    "AAPL": FinancialMetrics(
        revenue=385600.0,
        revenue_growth_yoy=8.5,
        net_income=101900.0,
        net_margin=26.4,
        ocf=118200.0,
        debt_to_equity=1.45,
        pe_ratio=32.4,
        eps=6.55,
        currency="USD"
    ),
    "MSFT": FinancialMetrics(
        revenue=245100.0,
        revenue_growth_yoy=15.2,
        net_income=88100.0,
        net_margin=35.9,
        ocf=110400.0,
        debt_to_equity=0.42,
        pe_ratio=34.8,
        eps=11.80,
        currency="USD"
    ),
    "NVDA": FinancialMetrics(
        revenue=120800.0,
        revenue_growth_yoy=122.0,
        net_income=65200.0,
        net_margin=54.0,
        ocf=68500.0,
        debt_to_equity=0.18,
        pe_ratio=55.2,
        eps=2.65,
        currency="USD"
    ),
    "DGC": FinancialMetrics(
        revenue=11200.0,
        revenue_growth_yoy=12.8,
        net_income=3450.0,
        net_margin=30.8,
        ocf=3920.0,
        debt_to_equity=0.12,
        pe_ratio=14.5,
        eps=8900.0,
        currency="VND"
    ),
    "VNM": FinancialMetrics(
        revenue=62500.0,
        revenue_growth_yoy=4.2,
        net_income=9600.0,
        net_margin=15.4,
        ocf=10800.0,
        debt_to_equity=0.28,
        pe_ratio=16.8,
        eps=4350.0,
        currency="Tỷ VND"
    ),
    "VCB": FinancialMetrics(
        revenue=68000.0,
        revenue_growth_yoy=10.5,
        net_income=33000.0,
        net_margin=48.5,
        ocf=25000.0,
        debt_to_equity=0.15,
        pe_ratio=13.5,
        eps=7500.0,
        currency="Tỷ VND"
    ),
    "FPT": FinancialMetrics(
        revenue=52600.0,
        revenue_growth_yoy=19.6,
        net_income=7800.0,
        net_margin=14.8,
        ocf=8200.0,
        debt_to_equity=0.25,
        pe_ratio=21.0,
        eps=6200.0,
        currency="Tỷ VND"
    ),
    "HPG": FinancialMetrics(
        revenue=140000.0,
        revenue_growth_yoy=16.0,
        net_income=12000.0,
        net_margin=8.6,
        ocf=15000.0,
        debt_to_equity=0.55,
        pe_ratio=12.0,
        eps=2100.0,
        currency="Tỷ VND"
    ),
    "KO": FinancialMetrics(
        revenue=45800.0,
        revenue_growth_yoy=5.8,
        net_income=10700.0,
        net_margin=23.4,
        ocf=11600.0,
        debt_to_equity=1.40,
        pe_ratio=24.5,
        eps=2.48,
        currency="USD"
    ),
    "TSLA": FinancialMetrics(
        revenue=96800.0,
        revenue_growth_yoy=7.8,
        net_income=7100.0,
        net_margin=7.3,
        ocf=8900.0,
        debt_to_equity=0.15,
        pe_ratio=68.5,
        eps=2.15,
        currency="USD"
    ),
}

