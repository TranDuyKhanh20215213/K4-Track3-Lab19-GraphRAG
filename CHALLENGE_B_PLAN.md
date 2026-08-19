# Plan: Entity Resolution Guard Improvements (AI Coding Agent Challenge B)

**Nguồn:** Notebook cell "2.2 — Entity resolution", markdown "🎯 AI Coding Agent Challenge B" — cải
tiến guard cho: ticker, suffix `Inc./Corp./Ltd.`, product chứa company name, người trùng họ/tên gần
giống. Trực tiếp map vào tiêu chí Module 3 trong `ASSIGNMENT.md`: "Ngăn chặn được các trường hợp
False Merge nguy hiểm (Sam Altman vs Steve Altman, Apple Watch vs Apple)."

**Vị trí:** Notebook `Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb`, cell id `223090d5`
("#@title 2.2 — Entity resolution"). Sửa function có sẵn, không tạo cell mới.

## Task 1 — Mở rộng `CORP_SUFFIXES`
Thêm biến thể phổ biến chưa có: `group`, `holdings`, `technologies`, `labs`, `sa`, `ag`, `gmbh`,
`nv`. Verify: grep xác nhận set mới trong notebook.

## Task 2 — Viết lại `merge_guard(a, b, typ=None)`
Trả về `(bool, reason)` thay vì `bool`. Rule theo thứ tự, rule đầu tiên khớp quyết định:

```python
def _tokens(name):
    return strip_suffix(name).split()

def _is_ticker_like(raw_name):
    core = re.sub(r"[^A-Za-z]", "", raw_name)
    return 1 <= len(core) <= 5 and core.isupper()

def _initials(tokens):
    return "".join(t[0] for t in tokens if t).upper()

def merge_guard(a, b, typ=None):
    na, nb = strip_suffix(a), strip_suffix(b)
    if na == nb:
        return True, "MERGE_EXACT_SUFFIX_STRIP"

    toks_a, toks_b = set(na.split()), set(nb.split())
    if toks_a and toks_b and toks_a != toks_b and (toks_a < toks_b or toks_b < toks_a):
        return False, "CONTAINMENT_MISMATCH"

    a_ticker, b_ticker = _is_ticker_like(a), _is_ticker_like(b)
    if a_ticker != b_ticker:
        ticker_raw, other_toks = (a, nb.split()) if a_ticker else (b, na.split())
        ticker_core = re.sub(r"[^A-Za-z]", "", ticker_raw).upper()
        if _initials(other_toks) == ticker_core:
            return True, "TICKER_INITIALS_MATCH"
        return False, "TICKER_UNVERIFIED"

    if typ == "Person":
        ta, tb = na.split(), nb.split()
        if ta and tb and ta[-1] == tb[-1] and ta[0] != tb[0]:
            if not (tb[0].startswith(ta[0]) or ta[0].startswith(tb[0])):
                return False, "PERSON_GIVEN_NAME_MISMATCH"

    ratio = SequenceMatcher(None, na, nb).ratio()
    if ratio >= 0.72:
        return True, "LEXICAL_RATIO_OK"
    return False, "LOW_LEXICAL_SIMILARITY"
```

Ghi chú thiết kế (đã thống nhất qua brainstorming):
- **Containment guard**: dùng tập con thực sự (proper subset) trên token-set sau suffix-strip —
  bắt "Apple" ⊂ "Apple Watch" nhưng không false-positive với các cặp identical.
- **Ticker guard**: heuristic "ticker-like" = chuỗi gốc (chưa lowercase) dài 1–5 ký tự, toàn chữ hoa.
  So khớp initials của tên đầy đủ kia; khớp thì merge có căn cứ rõ ràng (`TICKER_INITIALS_MATCH`),
  không khớp thì từ chối vì `SequenceMatcher` không đáng tin với chuỗi quá ngắn.
- **Person guard**: chỉ kích hoạt khi `typ=="Person"`. Từ chối khi họ (token cuối) trùng nhưng tên
  (token đầu) khác nhau, TRỪ KHI một bên là tiền tố của bên kia (cho phép "Sam"/"Samuel"). Quyết
  định đã chốt: strict-reject theo hướng ưu tiên precision, chấp nhận bỏ sót một số biến thể
  nickname không phải quan hệ tiền tố (VD "Bill"/"William") — đây là điểm cần nêu trong báo cáo.
- Rule cũ (SequenceMatcher ratio ≥ 0.72) giữ nguyên làm fallback cuối cùng, không đổi threshold.

## Task 3 — Cập nhật call site (dòng ~934-942)
```python
if j < 0 or i >= j or float(score) < threshold:
    continue
ok, guard_reason = merge_guard(names[i], names[j], typ=typ)
audit.append({
    "type": typ, "left": names[i], "right": names[j],
    "similarity": float(score),
    "decision": "MERGE_VECTOR" if ok else "REJECT_GUARD",
    "guard_reason": guard_reason,
})
if ok:
    uf.union(i, j)
```
Chỉ thêm cột `guard_reason` — cột `decision` giữ nguyên `MERGE_VECTOR`/`REJECT_GUARD` để
`show_resolution_audit()` ở Phần 5.1 (cell khác, không sửa) tiếp tục hoạt động không đổi.

## Task 4 — Kiểm tra tính hợp lệ notebook
- `json.load` xác nhận file vẫn hợp lệ.
- `ast.parse` trên source cell đã sửa để xác nhận cú pháp Python hợp lệ.

## Task 5 — Standalone unit test (venv cô lập, không commit script test)
Trích xuất cell thật từ notebook, chạy trực tiếp `merge_guard` với các case từ đề bài:
1. `merge_guard("IBM", "International Business Machines", typ="Company")` → `(True, "TICKER_INITIALS_MATCH")`.
2. `merge_guard("Apple", "Apple Watch", typ="Technology")` → `(False, "CONTAINMENT_MISMATCH")`.
3. `merge_guard("Sam Altman", "Steve Altman", typ="Person")` → `(False, "PERSON_GIVEN_NAME_MISMATCH")`.
4. `merge_guard("Sam Altman", "Samuel Altman", typ="Person")` → `True` (given-name tiền tố, không bị chặn nhầm).
5. `merge_guard("Foo Holdings", "Foo", typ="Company")` → `(True, "MERGE_EXACT_SUFFIX_STRIP")` (kiểm tra CORP_SUFFIXES mở rộng).
6. Case cũ vẫn đúng: `merge_guard("Microsoft Corp", "Microsoft Corporation", typ="Company")` → merge qua suffix-strip.

## Task 6 — Không tự động điền báo cáo
Không sửa `reports/lab_report.md`. Ghi lại 3 nội dung để học viên điền câu hỏi thuyết minh #2, #3
trong `ASSIGNMENT.md`:
1. **Threshold không đổi**: vẫn 0.72 cho fallback lexical ratio — Challenge B không thay ngưỡng mà
   thêm rule tường minh chạy trước ratio.
2. **Cặp similarity cao nhưng bị chặn**: sau khi chạy trên dữ liệu thật, lọc
   `entity_resolution_audit_df[decision=="REJECT_GUARD"]` sort theo `similarity` giảm dần, cột
   `guard_reason` cho biết lý do cụ thể (VD `CONTAINMENT_MISMATCH` hoặc `PERSON_GIVEN_NAME_MISMATCH`)
   thay vì phải suy đoán.
3. **Giới hạn đã biết**: rule Person không xử lý nickname không phải tiền tố (Bill/William,
   Bob/Robert) — sẽ bị reject dù có thể là cùng một người; đây là trade-off precision-over-recall
   có chủ đích, phù hợp triết lý "prefer precision" của lab.

## Out of scope
- Không đổi `MANUAL_ALIASES` hay logic `build_resolution_map`/`canonicalize_triples` ngoài việc
  truyền thêm `typ` và `guard_reason`.
- Không sửa `show_resolution_audit()` ở Phần 5.1 — audit mới tương thích ngược hoàn toàn.
- Không chạy toàn bộ notebook trên dữ liệu thật (cần Neo4j/API keys); chỉ verify cú pháp + unit test
  cục bộ trên các case nêu trong đề bài.
