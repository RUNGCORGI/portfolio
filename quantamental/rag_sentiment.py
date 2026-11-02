"""
quantamental/rag_sentiment.py — RAG Chấm điểm Cảm xúc Báo chí & Mức độ Sôi động Truyền thông (Flag S).
Tích hợp:
1. Mô hình Ngôn ngữ Tiếng Việt PhoBERT (wonrax/phobert-base-vietnamese-sentiment) & pyvi word segmenter
   cho thị trường Việt Nam (thay thế bộ từ khóa thủ công).
2. NLTK VADER (Valence Aware Dictionary and sEntiment Reasoner) và Financial Domain Lexicon cho thị trường Mỹ (US).
Phân tích dữ liệu phi cấu trúc từ các bài báo thời gian thực đa thị trường.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from quantamental.models import NewsItem, SentimentResult

# Thiết lập cờ môi trường tối ưu cho PyTorch và Hugging Face trên Windows
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. KHỞI TẠO NLTK VADER CHO THỊ TRƯỜNG MỸ (US EQUITIES)
# ==============================================================================
_sia: Optional[object] = None
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    try:
        _sia = SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        _sia = SentimentIntensityAnalyzer()

    # Cập nhật từ vựng chuyên ngành Tài chính Phố Wall vào VADER (Financial Domain Adaptation)
    _FINANCIAL_LEXICON = {
        # Tích cực tài chính
        "bullish": 2.2, "rally": 2.0, "surge": 2.2, "outperform": 2.5, "upgrade": 2.0,
        "beat": 1.8, "record": 2.0, "growth": 1.8, "milestone": 1.5, "dividend": 1.5,
        "overweight": 2.0, "breakout": 1.8, "upside": 1.8, "profit": 1.8,
        # Tiêu cực tài chính
        "bearish": -2.2, "slump": -2.2, "plunge": -2.5, "downgrade": -2.2, "lawsuit": -2.0,
        "investigation": -2.0, "default": -2.5, "debt": -1.5, "probe": -1.8, "scandal": -2.2,
        "underperform": -2.0, "loss": -2.0, "warning": -1.8, "selloff": -2.2
    }
    _sia.lexicon.update(_FINANCIAL_LEXICON)
except Exception as e:
    logger.warning("NLTK VADER không thể khởi tạo, sử dụng bộ Lexicon dự phòng: %s", e)
    _sia = None

# ==============================================================================
# 2. KHỞI TẠO MÔ HÌNH PHOBERT CHO TIẾNG VIỆT (LAZY SINGLETON THREAD-SAFE)
# ==============================================================================
_PHOBERT_MODEL_NAME = "wonrax/phobert-base-vietnamese-sentiment"
_phobert_tokenizer = None
_phobert_model = None
_phobert_attempted = False
_phobert_lock = threading.Lock()

# Tập mã cổ phiếu Việt Nam thường gặp để ưu tiên điều hướng PhoBERT
_VN_TICKER_SET = {
    "VIC", "VCB", "FPT", "HPG", "VHM", "VNM", "TCB", "MSN", "MWG",
    "SSI", "STB", "MBB", "ACB", "VPB", "GAS", "BID", "VRE", "SAB",
    "POW", "PLX", "VJC", "TPB", "HDB", "SHB", "GVR", "KDH", "VIB"
}
_VN_CHAR_REGEX = re.compile(r"[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]", re.IGNORECASE)


def _get_phobert():
    """Tải PhoBERT Lazy Singleton an toàn đa luồng (Thread-Safe) để khởi động tức thì."""
    global _phobert_tokenizer, _phobert_model, _phobert_attempted
    if _phobert_attempted and _phobert_model is not None:
        return _phobert_tokenizer, _phobert_model

    with _phobert_lock:
        if _phobert_attempted and _phobert_model is not None:
            return _phobert_tokenizer, _phobert_model

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            logger.info("Đang nạp mô hình Ngôn ngữ PhoBERT (%s)...", _PHOBERT_MODEL_NAME)
            try:
                # Ưu tiên đọc trực tiếp từ cache máy cục bộ để tăng tốc tối đa
                _phobert_tokenizer = AutoTokenizer.from_pretrained(_PHOBERT_MODEL_NAME, local_files_only=True)
                _phobert_model = AutoModelForSequenceClassification.from_pretrained(_PHOBERT_MODEL_NAME, local_files_only=True)
            except Exception:
                _phobert_tokenizer = AutoTokenizer.from_pretrained(_PHOBERT_MODEL_NAME)
                _phobert_model = AutoModelForSequenceClassification.from_pretrained(_PHOBERT_MODEL_NAME)

            _phobert_model.eval()
            _phobert_attempted = True
            logger.info("PhoBERT Vietnamese Sentiment Model đã nạp thành công.")
        except Exception as exc:
            logger.warning("Không thể nạp mô hình PhoBERT: %s. Chuyển sang cơ chế Lexicon Tiếng Việt dự phòng.", exc)
            _phobert_tokenizer = None
            _phobert_model = None
            _phobert_attempted = True

    return _phobert_tokenizer, _phobert_model


def is_vietnamese_context(ticker: str, news_items: List[NewsItem]) -> bool:
    """Xác định bài viết hoặc mã cổ phiếu thuộc ngữ cảnh Tiếng Việt."""
    clean_ticker = ticker.upper().replace(".VN", "").strip()
    if clean_ticker in _VN_TICKER_SET or ticker.upper().endswith(".VN"):
        return True
    sample_text = " ".join(f"{it.title} {it.snippet}" for it in news_items[:3])
    return bool(_VN_CHAR_REGEX.search(sample_text))


# ==============================================================================
# 3. TỪ KHÓA DỰ PHÒNG & CHỈ SỐ FOMO / HYPE TRUYỀN THÔNG
# ==============================================================================
POSITIVE_KEYWORDS = [
    "kỷ lục", "tăng trưởng", "bứt phá", "vượt đỉnh", "lợi nhuận cao", "mở rộng", "hợp tác",
    "đột phá", "thành công", "tiềm năng", "khả quan", "khuyến nghị mua", "overweight",
    "record", "growth", "surge", "beat", "rally", "outperform", "bullish", "expansion",
    "dividend", "jump", "milestone", "innovation", "upgrade", "breakout", "lãi lớn"
]

NEGATIVE_KEYWORDS = [
    "thua lỗ", "sụt giảm", "lao dốc", "thanh tra", "vi phạm", "khởi tố", "điều tra", "nợ xấu",
    "bán tháo", "cảnh báo", "hạ bậc", "rủi ro", "phá sản", "khủng hoảng", "khiếu nại",
    "loss", "decline", "slump", "investigation", "lawsuit", "downgrade", "bearish", "default",
    "debt", "scandal", "probe", "warning", "plunge", "hủy niêm yết", "úp bô", "kéo xả"
]

HYPE_KEYWORDS = [
    "cơn sốt", "cháy hàng", "bùng nổ", "ngút ngàn", "điên cuồng", "siêu cổ phiếu",
    "frenzy", "craze", "hyped", "parabolic", "insane", "to the moon"
]


# ==============================================================================
# 4. CHỨC NĂNG CHẤM ĐIỂM CẢM XÚC PHOBERT (VIETNAMESE NLP PIPELINE)
# ==============================================================================
def _analyze_with_phobert(
    news_items: List[NewsItem]
) -> Tuple[float, int, int, int, List[str], str]:
    """
    Sử dụng PhoBERT (wonrax/phobert-base-vietnamese-sentiment) để phân loại cảm xúc tiếng Việt.
    Nhãn PhoBERT:
        0: NEG (Tiêu cực)
        1: POS (Tích cực)
        2: NEU (Trung tính)
    """
    tokenizer, model = _get_phobert()
    if tokenizer is None or model is None:
        raise RuntimeError("PhoBERT model not available")

    import torch
    try:
        from pyvi import ViTokenizer
    except ImportError:
        ViTokenizer = None

    pos_count = 0
    neg_count = 0
    hype_count = 0
    article_scores: List[float] = []
    prob_details: List[Dict[str, float]] = []
    found_themes: List[str] = []

    # Xử lý theo mini-batch (batch_size=16) để tăng tốc tối đa khi xử lý 100 bài báo
    batch_size = 16
    for i in range(0, len(news_items), batch_size):
        batch_items = news_items[i : i + batch_size]
        batch_texts = []
        for it in batch_items:
            t = f"{it.title}. {it.snippet}".strip()
            batch_texts.append(ViTokenizer.tokenize(t) if ViTokenizer else t)

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

        for j, it in enumerate(batch_items):
            p_neg = float(probs[j, 0].item())
            p_pos = float(probs[j, 1].item())
            p_neu = float(probs[j, 2].item())

            prob_details.append({"pos": p_pos, "neg": p_neg, "neu": p_neu})

            # Phân loại nhãn theo ngưỡng tin cậy của PhoBERT
            if (p_pos >= 0.45) or (p_pos > p_neg and p_pos > 0.30):
                pos_count += 1
            elif (p_neg >= 0.45) or (p_neg > p_pos and p_neg > 0.30):
                neg_count += 1

            # Chấm điểm liên tục: 0.50 cơ sở + 0.45 * (P(POS) - P(NEG))
            item_score = 0.50 + 0.45 * (p_pos - p_neg)
            article_scores.append(item_score)

            text_lower = f"{it.title} {it.snippet}".lower()
            if any(k in text_lower for k in HYPE_KEYWORDS):
                hype_count += 1

    total_articles = len(news_items)
    avg_score = sum(article_scores) / max(total_articles, 1)

    # Thêm yếu tố Hype Metric nếu có hiện tượng FOMO truyền thông
    if hype_count > 0:
        avg_score += 0.08
        found_themes.append("Xuất hiện yếu tố FOMO/thổi phồng từ truyền thông (Hype Metric)")

    final_score = round(max(0.05, min(0.95, avg_score)), 2)

    avg_pos = sum(p["pos"] for p in prob_details) / len(prob_details)
    avg_neg = sum(p["neg"] for p in prob_details) / len(prob_details)
    avg_neu = sum(p["neu"] for p in prob_details) / len(prob_details)

    reasoning_detail = (
        f"Mô hình Ngôn ngữ PhoBERT AI: {pos_count} bài tích cực, {neg_count} bài tiêu cực "
        f"(Xác suất trung bình: POS {avg_pos:.1%}, NEG {avg_neg:.1%}, NEU {avg_neu:.1%})"
    )

    return final_score, pos_count, neg_count, hype_count, found_themes, reasoning_detail


# ==============================================================================
# 5. HÀM ĐIỀU PHỐI CHÍNH (ANALYZE SENTIMENT)
# ==============================================================================
def analyze_sentiment(ticker: str, news_items: List[NewsItem]) -> SentimentResult:
    """
    RAG Agent phân tích cảm xúc từ danh sách tin tức thời gian thực của mã cổ phiếu.
    - Với thị trường Việt Nam: Ưu tiên mô hình Deep Learning PhoBERT (wonrax/phobert-base-vietnamese-sentiment).
    - Với thị trường Mỹ: Sử dụng NLTK VADER với Financial Domain Adaptation.
    - Dự phòng an toàn: Bộ từ điển Lexicon chuyên sâu.
    """
    if not news_items:
        return SentimentResult(
            ticker=ticker,
            score=0.50,
            flag=0,
            label="Trung tính (Thiếu tin)",
            news_items=[],
            key_themes=["Không có tin tức đáng kể gần đây"],
            reasoning="Chưa có dữ liệu bài báo mới để đánh giá cảm xúc, mặc định giữ trạng thái trung tính."
        )

    is_vn = is_vietnamese_context(ticker, news_items)
    total_articles = len(news_items)

    # --------------------------------------------------------------------------
    # TRƯỜNG HỢP A: THỊ TRƯỜNG VIỆT NAM -> SỬ DỤNG PHOBERT
    # --------------------------------------------------------------------------
    if is_vn:
        try:
            score, pos_count, neg_count, hype_count, found_themes, phobert_detail = _analyze_with_phobert(news_items)
            flag_s = 1 if (score >= 0.58 and pos_count >= neg_count) else 0

            if score >= 0.65:
                label = "Tích cực / Sôi động"
                found_themes.append("PhoBERT ghi nhận làn sóng tin tích cực về kết quả kinh doanh & triển vọng")
            elif score <= 0.40:
                label = "Tiêu cực / Cảnh báo"
                found_themes.append("PhoBERT ghi nhận áp lực rủi ro, nợ xấu hoặc lo ngại sụt giảm")
            else:
                label = "Trung tính / Bình thường"
                found_themes.append("PhoBERT nhận định luồng tin cân bằng, thị trường chưa có đột biến tâm lý")

            reasoning = (
                f"Phân tích {total_articles} bài báo bằng {phobert_detail}. "
                f"Điểm cảm xúc PhoBERT đo được là {score:.2f} "
                f"({'ĐẠT' if flag_s == 1 else 'CHƯA ĐẠT'} ngưỡng sôi động truyền thông S=1)."
            )

            return SentimentResult(
                ticker=ticker,
                score=score,
                flag=flag_s,
                label=label,
                news_items=news_items,
                key_themes=found_themes,
                reasoning=reasoning
            )
        except Exception as err:
            logger.warning("PhoBERT không khả dụng (%s), chuyển sang Lexicon Tiếng Việt dự phòng.", err)

    # --------------------------------------------------------------------------
    # TRƯỜNG HỢP B: THỊ TRƯỜNG MỸ HOẶC DỰ PHÒNG LEXICON
    # --------------------------------------------------------------------------
    pos_count = 0
    neg_count = 0
    hype_count = 0
    vader_compounds: List[float] = []
    found_themes = []

    for item in news_items:
        text = f"{item.title} {item.snippet}"
        text_lower = text.lower()

        # Chấm điểm NLTK VADER
        vader_compound = 0.0
        if _sia is not None:
            try:
                vader_res = _sia.polarity_scores(text)
                vader_compound = float(vader_res.get("compound", 0.0))
                vader_compounds.append(vader_compound)
            except Exception:
                pass

        # Nhận diện tín hiệu phân cực
        has_pos = (vader_compound >= 0.15) or any(k in text_lower for k in POSITIVE_KEYWORDS)
        has_neg = (vader_compound <= -0.15) or any(k in text_lower for k in NEGATIVE_KEYWORDS)
        has_hype = any(k in text_lower for k in HYPE_KEYWORDS)

        if has_pos:
            pos_count += 1
        if has_neg:
            neg_count += 1
        if has_hype:
            hype_count += 1

    net_diff = (pos_count - neg_count) / max(total_articles, 1)
    base_score = 0.50 + 0.40 * net_diff

    if vader_compounds:
        avg_vader = sum(vader_compounds) / len(vader_compounds)
        if abs(avg_vader) > 0.05:
            base_score = 0.50 + 0.30 * net_diff + 0.10 * avg_vader

    if hype_count > 0:
        base_score += 0.10
        found_themes.append("Xuất hiện yếu tố FOMO/thổi phồng từ truyền thông (Hype Metric)")

    score = round(max(0.05, min(0.95, base_score)), 2)
    flag_s = 1 if (score >= 0.58 and pos_count >= neg_count) else 0

    engine_name = "NLTK VADER & Wall St. Lexicon" if not is_vn else "Financial Domain Lexicon"
    if score >= 0.65:
        label = "Tích cực / Sôi động"
        found_themes.append(f"{engine_name} ghi nhận làn sóng tin tích cực về triển vọng tăng trưởng")
    elif score <= 0.40:
        label = "Tiêu cực / Cảnh báo"
        found_themes.append(f"{engine_name} phản ánh quan ngại về rủi ro hoặc áp lực điều chỉnh")
    else:
        label = "Trung tính / Bình thường"
        found_themes.append("Báo chí đưa tin cân bằng, thị trường chưa có làn sóng FOMO")

    vader_info = f" (VADER compound: {sum(vader_compounds)/len(vader_compounds):.2f})" if vader_compounds else ""
    reasoning = (
        f"Phân tích {total_articles} bài báo bằng {engine_name}: ghi nhận {pos_count} bài tích cực, "
        f"{neg_count} bài tiêu cực{vader_info}. Điểm cảm xúc đo được là {score:.2f} "
        f"({'ĐẠT' if flag_s == 1 else 'CHƯA ĐẠT'} ngưỡng sôi động truyền thông S=1)."
    )

    return SentimentResult(
        ticker=ticker,
        score=score,
        flag=flag_s,
        label=label,
        news_items=news_items,
        key_themes=found_themes,
        reasoning=reasoning
    )
