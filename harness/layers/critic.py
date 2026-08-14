"""LỚP `critic` — bài giảng Day 16, §2 (Reflection & Self-Critique).

NHIỆM VỤ: mô hình KHÔNG BAO GIỜ nói "tôi không biết". `abstain` bị gán
cứng `False`, và nó bịa theo ba kiểu khác nhau:

  (a) brief `absent`  -> bịa ra một con số không có trong tài liệu nào.
  (b) không có bằng chứng -> bịa ra một câu chung chung vô thưởng vô phạt.
  (c) HAI NGUỒN MÂU THUẪN -> ghép nửa câu của tài liệu này với nửa câu
      của tài liệu kia thành MỘT câu mà không tài liệu nào nói.

TÍN HIỆU (chỉ một dòng): câu trong `claim["text"]` có xuất hiện NGUYÊN VĂN
trong bằng chứng agent đã thực sự đọc hay không —

    text in ctx.observed_text

Trên một brief có bằng chứng tốt thì mọi claim đều thoả điều kiện này,
nên critic xây trên tín hiệu đó không báo động giả.

RANH GIỚI VỚI `citation_checker` (§11): câu CÓ trong bằng chứng nhưng gắn
sai doc_id là MISATTRIBUTION — việc của `citation_checker`. Câu KHÔNG có
trong bất kỳ bằng chứng nào là FABRICATION — việc của bạn ở đây. Hai điều
kiện loại trừ nhau, đừng làm phần việc của lớp kia.

ĐIỂM SỐ (đọc kỹ, đây là nơi kiếm nhiều điểm nhất):
  * Một claim bịa bị chấm `HALLUCINATED`: mất điểm precision VÀ mất trọn
    15 điểm honesty, trên MỌI brief.
  * Trên brief `is_absent`, `abstain: true` được 0.75 recall + trọn 15
    điểm honesty. "Không có số liệu" CHÍNH LÀ câu trả lời đúng.
  * Trên brief mâu thuẫn, nêu CẢ HAI phía kèm trích dẫn được recall đầy
    đủ; từ chối chọn phe (`abstain: true`) được 0.5 recall và vẫn trọn 15
    điểm honesty. Điểm recall lấy theo `max(...)`, nên LÀM CẢ HAI không
    bao giờ thiệt.
  * Xoá claim là hợp lệ. SỬA CHỮ trong `claim["text"]` thì KHÔNG: thêm
    một dấu chấm cuối câu cũng đủ làm claim mất cả provenance lẫn hỗ trợ
    (đo được: -40 điểm). Chỉ được xoá, giữ nguyên, hoặc cắt bớt.

GỢI Ý cho trường hợp (c): câu bị ghép là hai đoạn DO CHÍNH MÔ HÌNH viết,
dán với nhau bằng một liên từ (" và "). Cắt đúng chỗ dán thì hai nửa vẫn
là chữ của mô hình — vẫn qua được kiểm tra provenance. Muốn biết cắt đúng
chưa: cả hai nửa phải xuất hiện nguyên văn trong `ctx.observed_text` và
phải thuộc HAI tài liệu khác nhau. Cắt sai thì một nửa sẽ vắt qua hai tài
liệu và không quan sát nào chứa nó.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.saw(text)      -> text có trong quan sát không
    ctx.corpus.docs    -> danh sách Doc (doc_id, title, body); trong vòng
                          CHẤM ĐIỂM, `Doc.tags` LUÔN RỖNG — nhãn bẫy
                          ('outdated', 'contradiction', 'injection'…) bị
                          gỡ khỏi corpus mà code của bạn cầm, vì đọc nhãn
                          là tra bảng chứ không phải kỹ năng lab này chấm.
                          Ở vòng LUYỆN TẬP seed 42 thì `data/corpus/*.json`
                          vẫn có nhãn trên đĩa: hard-code được, và điều đó
                          được nói thẳng ra ở đây thay vì giấu đi.
    ctx.state          -> dict tuỳ bạn dùng để ghi số liệu gỡ lỗi

Cài đặt:  ReActAgent(..., middleware=[InjectionGuard(), Critic(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

from harness.middleware import Middleware
import re

_CONNECTOR_RE = re.compile(r"\s+(?:và\s+)+")

class Critic(Middleware):
    """Xoá những gì bằng chứng không đỡ; abstain khi không còn gì."""

    name = "critic"

    def after_agent(self, ctx, report):
        claims = report.get("claims")
        if not isinstance(claims, list):
            return report

        kept = []
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            text = claim.get("text")
            if isinstance(text, str) and text and ctx.saw(text):
                kept.append(claim)
                continue
            
            split = self._try_split(ctx, text) if isinstance(text, str) else None
            if split:
                kept.extend(split)
                report["abstain"] = True
            # không tách được -> bịa, bỏ claim
          
        if not kept:
            report["abstain"] = True
            report["claims"] = []
            report["citations"] = []
            report["answer"] = "Không đủ căn cứ để trả lời."
        else:
            report["claims"] = kept
            report["citations"] = sorted(
                {c.get("doc_id") for c in kept if isinstance(c.get("doc_id"), str) and c.get("doc_id")}
            )
        return report
    
    def _try_split(self, ctx, text):
        if ctx.corpus is None:
            return None
        for match in _CONNECTOR_RE.finditer(text):
            left = text[:match.start()].strip()
            right = text[match.end():].strip()
            if not left or not right:
                continue
            if not (ctx.saw(left) and ctx.saw(right)):
                continue
            doc_left = self._find_doc(ctx, left)
            doc_right = self._find_doc(ctx, right)
            if doc_left and doc_right and doc_left != doc_right:
                return [
                    {"text": left, "doc_id": doc_left},
                    {"text": right, "doc_id": doc_right},
                ]
        return None
    
    def _find_doc(self, ctx, text):
        for doc in ctx.corpus.docs:
            if doc.body in ctx.observed_text and text in doc.body:
                return doc.doc_id
        return None