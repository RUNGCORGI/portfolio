"""Xuất báo cáo HTML và PDF có khả năng tái lập.

Khi nộp bài hoặc gửi kết quả cho người khác, ảnh chụp màn hình dashboard là
bằng chứng yếu: không ai biết bạn chạy với tham số gì, dữ liệu tới ngày nào.

Module này sinh báo cáo kèm **mã tái lập (run hash)** — một chuỗi băm của
toàn bộ cấu hình và dữ liệu đầu vào. Hai báo cáo cùng mã băm chắc chắn đến
từ cùng một lần chạy; khác mã băm nghĩa là có thứ gì đó đã thay đổi.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Dấu vết lần chạy
# --------------------------------------------------------------------------


@dataclass
class RunManifest:
    """Toàn bộ thông tin cần để tái lập đúng một lần chạy."""

    tickers: list[str]
    start: str
    end: str
    method: str
    config: dict[str, Any] = field(default_factory=dict)
    data_rows: int = 0
    data_last_date: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def run_hash(self) -> str:
        """Mã băm ổn định của cấu hình, không phụ thuộc thứ tự khai báo."""
        payload = json.dumps(
            {
                "tickers": sorted(self.tickers),
                "start": self.start,
                "end": self.end,
                "method": self.method,
                "config": self.config,
                "data_rows": self.data_rows,
                "data_last_date": self.data_last_date,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def to_table(self) -> pd.DataFrame:
        d = asdict(self)
        d["tickers"] = ", ".join(self.tickers)
        d["config"] = json.dumps(self.config, ensure_ascii=False)
        names = {
            "tickers": "Danh mục mã", "start": "Từ ngày", "end": "Đến ngày",
            "method": "Phương pháp", "config": "Cấu hình", "data_rows": "Số phiên dữ liệu",
            "data_last_date": "Dữ liệu tới ngày", "created_at": "Thời điểm tạo",
        }
        rows = [{"Mục": names.get(k, k), "Giá trị": v} for k, v in d.items()]
        rows.append({"Mục": "Mã tái lập", "Giá trị": self.run_hash})
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Khối nội dung
# --------------------------------------------------------------------------


@dataclass
class Section:
    """Một mục trong báo cáo: tiêu đề, diễn giải, bảng và biểu đồ."""

    title: str
    commentary: str = ""
    table: pd.DataFrame | None = None
    figure_png: bytes | None = None
    note: str = ""


def figure_to_png(fig) -> bytes | None:
    """Chuyển biểu đồ Plotly hoặc Matplotlib thành PNG.

    Trả về ``None`` nếu môi trường thiếu bộ xuất ảnh — báo cáo vẫn sinh được
    với bảng số, chỉ thiếu hình.
    """
    try:
        if hasattr(fig, "to_image"):
            return fig.to_image(format="png", width=1000, height=500, scale=2)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        return buf.getvalue()
    except Exception:
        return None


def series_to_png(data: pd.DataFrame | pd.Series, title: str = "") -> bytes | None:
    """Vẽ nhanh một biểu đồ đường bằng Matplotlib rồi trả về PNG.

    Dùng Matplotlib thay vì Plotly vì nó không cần Chrome/Kaleido, nên luôn
    chạy được trong Docker hoặc máy chấm bài.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4.2))
        df = data.to_frame() if isinstance(data, pd.Series) else data
        for col in df.columns:
            ax.plot(df.index, df[col], label=str(col), linewidth=1.4)
        ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.25)
        if df.shape[1] > 1:
            ax.legend(fontsize=9)
        fig.tight_layout()
        out = figure_to_png(fig)
        plt.close(fig)
        return out
    except Exception:
        return None


# --------------------------------------------------------------------------
# Kết xuất HTML
# --------------------------------------------------------------------------

_CSS = """
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;margin:0;padding:40px;
color:#1b1f23;background:#fff;line-height:1.55}
.wrap{max-width:980px;margin:0 auto}
h1{font-size:26px;margin:0 0 4px;border-bottom:3px solid #0b63c5;padding-bottom:10px}
h2{font-size:18px;margin:34px 0 8px;color:#0b63c5}
.sub{color:#5a6472;font-size:13px;margin-bottom:22px}
.hash{display:inline-block;background:#eef4fc;border:1px solid #c6dcf5;border-radius:5px;
padding:3px 9px;font-family:Consolas,monospace;font-size:12px;color:#0b4f9e}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}
th{background:#f3f6fa;text-align:left;padding:8px 10px;border-bottom:2px solid #d4dde8;font-weight:600}
td{padding:7px 10px;border-bottom:1px solid #eceff3}
tr:nth-child(even) td{background:#fafbfd}
.num{text-align:right;font-variant-numeric:tabular-nums}
.cmt{background:#f7f9fc;border-left:3px solid #0b63c5;padding:10px 14px;margin:10px 0;
font-size:13px;color:#33404f}
.note{font-size:12px;color:#6b7684;font-style:italic;margin-top:6px}
img{max-width:100%;border:1px solid #e3e8ee;border-radius:6px;margin:10px 0}
footer{margin-top:44px;padding-top:14px;border-top:1px solid #e3e8ee;font-size:11px;color:#8a94a0}
"""


def _fmt_df(df: pd.DataFrame) -> str:
    """Chuyển DataFrame thành HTML, số căn phải và làm tròn gọn."""
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: f"{v:,.4f}" if pd.notna(v) else "")
    html = d.to_html(index=False, escape=False, border=0)
    return html.replace('<td>', '<td class="num">').replace(
        '<td class="num">', '<td class="num">', 1
    )


def build_html(
    title: str,
    manifest: RunManifest,
    sections: list[Section],
    subtitle: str = "Báo cáo phân tích danh mục đầu tư",
) -> str:
    """Dựng báo cáo HTML hoàn chỉnh, một file duy nhất, ảnh nhúng sẵn."""
    parts = [
        "<!DOCTYPE html><html lang='vi'><head><meta charset='utf-8'>",
        f"<title>{title}</title><style>{_CSS}</style></head><body><div class='wrap'>",
        f"<h1>{title}</h1>",
        f"<div class='sub'>{subtitle} &nbsp;·&nbsp; "
        f"Tạo lúc {manifest.created_at} &nbsp;·&nbsp; "
        f"<span class='hash'>Mã tái lập: {manifest.run_hash}</span></div>",
    ]

    parts.append("<h2>Thông số lần chạy</h2>")
    parts.append(_fmt_df(manifest.to_table()))

    for sec in sections:
        parts.append(f"<h2>{sec.title}</h2>")
        if sec.commentary:
            parts.append(f"<div class='cmt'>{sec.commentary}</div>")
        if sec.figure_png:
            b64 = base64.b64encode(sec.figure_png).decode()
            parts.append(f"<img src='data:image/png;base64,{b64}' alt='{sec.title}'/>")
        if sec.table is not None and not sec.table.empty:
            parts.append(_fmt_df(sec.table))
        if sec.note:
            parts.append(f"<div class='note'>{sec.note}</div>")

    parts.append(
        "<footer>Báo cáo sinh tự động từ Quant Lab. Nội dung chỉ phục vụ mục đích "
        "nghiên cứu và học tập, không phải khuyến nghị đầu tư. Kết quả quá khứ "
        "không đảm bảo kết quả tương lai.</footer></div></body></html>"
    )
    return "\n".join(parts)


def save_html(html: str, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# Kết xuất PDF
# --------------------------------------------------------------------------


def html_to_pdf(html: str, path: str | Path) -> Path | None:
    """Chuyển HTML sang PDF bằng WeasyPrint nếu có."""
    try:
        from weasyprint import HTML

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        HTML(string=html).write_pdf(str(p))
        return p
    except Exception:
        return None


def build_pdf_fallback(
    title: str, manifest: RunManifest, sections: list[Section], path: str | Path
) -> Path | None:
    """Sinh PDF bằng ReportLab khi WeasyPrint không dùng được.

    Giữ lại đầy đủ bảng số và diễn giải, chỉ đơn giản hơn về trình bày.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
        )

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(
            str(p), pagesize=A4,
            leftMargin=18 * mm, rightMargin=18 * mm,
            topMargin=16 * mm, bottomMargin=16 * mm,
        )
        ss = getSampleStyleSheet()
        h1 = ParagraphStyle("h1x", parent=ss["Heading1"], fontSize=17, spaceAfter=4)
        h2 = ParagraphStyle("h2x", parent=ss["Heading2"], fontSize=12,
                            textColor=colors.HexColor("#0b63c5"), spaceBefore=12)
        body = ParagraphStyle("bodyx", parent=ss["BodyText"], fontSize=8.5, leading=12)

        flow = [
            Paragraph(title, h1),
            Paragraph(
                f"Tạo lúc {manifest.created_at} &nbsp;|&nbsp; "
                f"Mã tái lập: <b>{manifest.run_hash}</b>", body,
            ),
            Spacer(1, 8),
            _rl_table(manifest.to_table(), Table, TableStyle, colors),
        ]

        for sec in sections:
            flow.append(Paragraph(sec.title, h2))
            if sec.commentary:
                flow.append(Paragraph(sec.commentary, body))
                flow.append(Spacer(1, 4))
            if sec.figure_png:
                try:
                    flow.append(Image(io.BytesIO(sec.figure_png), width=165 * mm, height=70 * mm))
                    flow.append(Spacer(1, 4))
                except Exception:
                    pass
            if sec.table is not None and not sec.table.empty:
                flow.append(_rl_table(sec.table, Table, TableStyle, colors))
            if sec.note:
                flow.append(Paragraph(f"<i>{sec.note}</i>", body))

        flow.append(Spacer(1, 14))
        flow.append(Paragraph(
            "<i>Báo cáo phục vụ nghiên cứu và học tập, không phải khuyến nghị đầu tư.</i>",
            body,
        ))
        doc.build(flow)
        return p
    except Exception:
        return None


def _rl_table(df: pd.DataFrame, Table, TableStyle, colors):
    """Dựng bảng ReportLab từ DataFrame."""
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: f"{v:,.4f}" if pd.notna(v) else "")
    data = [[str(c) for c in d.columns]] + d.astype(str).values.tolist()
    # cắt bớt ô quá dài để bảng không tràn trang
    data = [[(x[:46] + "…") if len(x) > 47 else x for x in row] for row in data]

    t = Table(data, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f6fa")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1b1f23")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#dde3ea")),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafbfd")]),
    ]))
    return t


def export_report(
    title: str,
    manifest: RunManifest,
    sections: list[Section],
    out_dir: str | Path = "reports",
    basename: str | None = None,
) -> dict[str, Path | None]:
    """Xuất cả HTML và PDF trong một lần gọi.

    Tên file có kèm mã tái lập, nên không bao giờ ghi đè nhầm lần chạy khác.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = basename or f"bao-cao-danh-muc-{manifest.run_hash}"

    html = build_html(title, manifest, sections)
    html_path = save_html(html, out_dir / f"{stem}.html")

    pdf_path = html_to_pdf(html, out_dir / f"{stem}.pdf")
    if pdf_path is None:
        pdf_path = build_pdf_fallback(title, manifest, sections, out_dir / f"{stem}.pdf")

    return {"html": html_path, "pdf": pdf_path}
