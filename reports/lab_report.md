# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Trần Duy Khánh
**MSSV:** 2A202601696
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 20/08/2026 · **Cập nhật theo lần chạy pipeline mới nhất:** 21/08/2026

> *Ghi chú nguồn dữ liệu:* Báo cáo này được viết lại hoàn toàn theo lần chạy pipeline mới nhất, tổng hợp từ: `outputs/graphrag_eval_results.csv` (25 câu hỏi Golden Dataset `G5000-26` → `G5000-50`, mỗi câu có câu trả lời + điểm chấm LLM-as-a-Judge cho cả Flat RAG và GraphRAG), `outputs/graphrag_vs_flatrag_summary.csv` (tổng hợp theo nhóm câu hỏi), `outputs/report_facts.md` (số liệu pipeline trích tự động), cùng các file audit chi tiết: `entity_resolution_audit.csv`, `top_degree_nodes.csv` / `top3_supernodes.csv`, `failure_cases.csv`, `coref_spotcheck.csv` + `cache_coref_*.csv`, `near_dedup_audit.csv`, `self_correction_routes.csv`, `community_reports.csv`, `golden_evidence_coverage.csv`. Các thông số kỹ thuật (ngưỡng, cơ chế guard, giới hạn super-node) được đối chiếu trực tiếp với code trong notebook `Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb`.
>
> **Thay đổi phạm vi dữ liệu so với lần chạy trước:** pipeline hiện chạy trên **5.000 dòng đầu** của `hackernoon_subset.csv` (khớp đúng scope của Golden Dataset), cho ra **2.118 bài báo** sau exact-dedup, **400 chunk** được đưa qua LLM extraction, tạo **287 node / 174 edge** trong đồ thị. Do phạm vi dữ liệu nhỏ hơn nhiều so với bản stream đầy đủ trước đây, nhiều con số thực nghiệm (top-degree, benchmark, ca lỗi) đã đổi khác hoàn toàn — báo cáo dưới đây bám sát 100% số liệu của lần chạy này, không tái sử dụng số liệu cũ.

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution (Phân giải đại từ)

*Trả lời:*
- **Kết quả kiểm tra thực tế:** Đối chiếu `outputs/coref_spotcheck.csv` (20 chunk lấy mẫu ngẫu nhiên) và toàn bộ cache `outputs/cache_coref_6e79b80e30e2c60e.csv` (400 chunk đã qua `resolve_coref_batch()`, cell 1.7), **không có chunk nào** có `unresolved_mentions` khác rỗng, và tất cả các phép thay thế đại từ quan sát được (ví dụ `r00043::c0000`: *"it unveiled its latest innovations"* → *"Samsung Electronics Co. Ltd. unveiled Samsung Electronics Co. Ltd.'s latest innovations"*) đều đúng vì mỗi chunk chỉ có **một** antecedent khả dĩ. Đây là bằng chứng cho thấy nguyên tắc precision-first (chỉ resolve khi rõ ràng, còn lại giữ nguyên) hoạt động đúng thiết kế trên tập dữ liệu này.
- **Tình huống rủi ro thật sự tồn tại trong dữ liệu (nơi cơ chế *có thể* phân giải sai nếu văn bản dài hơn):** Chunk `r03380::c0000` — evidence gốc của câu hỏi `G5000-29`/`G5000-33`/`G5000-30`: *"Seven tech companies including Google, Meta and OpenAI have voluntarily made commitments on developing and managing artificial intelligence."* Đây là kiểu câu có **3 công ty đồng xuất hiện** trong cùng một mệnh đề — đúng kịch bản mà `COREF_SYSTEM` được thiết kế để phòng ngừa. Trong bản chunk hiện tại không có đại từ mơ hồ cần resolve nên `coref_changed=False`, nhưng nếu bài báo gốc dài hơn và có câu tiếp theo dùng "the company said..." thì cả 3 antecedent (Google/Meta/OpenAI) đều hợp lệ về mặt cú pháp trong cùng chunk — lúc đó quy tắc "chỉ resolve khi antecedent rõ ràng trong cùng chunk" của hệ thống sẽ không còn thỏa mãn, và thiết kế đúng là phải bỏ ngỏ (log `unresolved_mentions`) thay vì đoán ngẫu nhiên một trong ba.
- **Hậu quả đối với Graph nếu rủi ro xảy ra:** Nếu một pipeline kém an toàn hơn đoán đại từ này về sai công ty (ví dụ gán cho Meta thay vì OpenAI), bước NER/RE (`run_extraction`, cell 2.1) sẽ tạo ra một **False Edge** kiểu `Meta -COMMITTED_TO-> AI Management` thay vì đúng chủ thể, làm sai lệch chính xác loại câu hỏi phân biệt vai trò theo công ty (`G5000-29`: mở rộng danh sách công ty cam kết White House; `G5000-33`: phân biệt sự kiện AP-OpenAI với cam kết White House). Thú vị là trong lần chạy này, lỗi cùng bản chất (nhầm thực thể do đồng xuất hiện dày đặc) **thực sự đã xảy ra** — nhưng ở một tầng khác của pipeline (Entity Resolution, xem mục 2) chứ không phải ở Coreference — cho thấy rủi ro "đồng xuất hiện nhiều thực thể trong cùng ngữ cảnh" là một failure mode xuyên suốt nhiều module, không riêng gì coref.

---

### 2. Entity Resolution Threshold & Lexical Guard

*Trả lời:*
- **Ngưỡng cấu hình:** cosine similarity `threshold = 0.90` (FAISS `IndexFlatIP` trên embedding đã normalize, `top_k = 5`), Lexical Guard `merge_guard()` yêu cầu `SequenceMatcher(...).ratio() >= 0.72` sau khi `strip_suffix()` loại các hậu tố công ty (`inc, corp, ltd, llc, plc, co, company,...`). Trên 26 dòng audit (`entity_resolution_audit.csv`), phân bố quyết định là: `REJECT_GUARD` 13, `REJECT_BELOW_THRESHOLD` 5, `MERGE_LEXICAL` 4, `MERGE_MANUAL` 3, `MERGE_VECTOR` 1.
- **Cặp thực thể similarity cao (> 0.85) bị Guard chặn đúng:** `Middlebury Institute community` vs `Middlebury Institute` — cosine similarity **0.9254** nhưng bị `REJECT_GUARD` với lý do `REJECT_SUBBRAND_CONTAINMENT`. Sau `strip_suffix()`, hai chuỗi chuẩn hoá thành `"middlebury institute community"` và `"middlebury institute"` — không trùng nhau vì `"community"` không nằm trong `CORP_SUFFIXES`; tỉ lệ ký tự khác biệt đủ lớn để `SequenceMatcher` cho ratio dưới 0.72, nên bị từ chối dù embedding coi hai tên là gần như đồng nghĩa (cùng chủ đề học thuật). **Lý do đây là quyết định đúng:** *"Middlebury Institute"* là tổ chức, còn *"Middlebury Institute community"* chỉ một tập hợp con (cộng đồng liên quan) — gộp chung sẽ làm mất phân biệt giữa phát ngôn chính thức của tổ chức và hoạt động của cộng đồng xung quanh nó, đúng tinh thần Lexical Guard mà đề bài minh hoạ bằng ví dụ `Google` vs `Google Cloud`.
- **Rủi ro ngược lại — một cặp KHÔNG nên gộp nhưng đã lọt qua Guard:** `NASDAQ: AAPL` vs `NASDAQ: META` (type `Product`), cosine similarity chỉ **0.6590** (dưới hẳn ngưỡng vector 0.90) nhưng vẫn được ghi nhận `MERGE_LEXICAL` với lý do `LEXICAL_RATIO_0.73` — tức là bị gộp qua nhánh xét chuỗi thuần túy (cả hai đều có dạng `"NASDAQ: XXXX"` nên tỉ lệ ký tự trùng nhau tình cờ vượt 0.72), dù `AAPL` (Apple) và `META` (Meta) là hai mã cổ phiếu của **hai công ty hoàn toàn khác nhau**. Đây là một False Merge thật sự đã xảy ra trong lần chạy này, và là nguyên nhân gốc rễ trực tiếp của ca lỗi `G5000-30` phân tích ở mục 4 — Lexical Guard hiệu quả với cặp tên dài (đã thấy ở case Middlebury) nhưng lại dễ bị đánh lừa bởi các chuỗi ngắn có cấu trúc mẫu lặp lại (`NASDAQ: <ticker>`), vì tỉ lệ ký tự giống nhau của một template cố định luôn cao bất kể ticker khác nhau.

---

### 3. Đồ thị & Super-node Mitigation

*Trả lời:*

| Hạng | Tên thực thể | Loại thực thể (Type) | Degree thực đo |
|------|--------------|---------------------|----------------------|
| 1 | Amazon | Company | 6 |
| 2 | ServiceNow | Company | 6 |
| 3 | L&T Technology Services | Company | 5 |

*(Nguồn: `outputs/top_degree_nodes.csv` / `top3_supernodes.csv`, đo trên đồ thị 287 node / 174 edge sau khi bulk-insert 400 chunk trích xuất.)*

- **Phát hiện quan trọng về quy mô:** Với `SUPER_NODE_DEGREE = 100` (cell 3.3), **không node nào trong lần chạy này thực sự vượt ngưỡng super-node** — degree cao nhất đo được (6) thấp hơn ngưỡng kích hoạt cap tới hơn 16 lần. Điều này khớp với cột `graph_supernode_events` trong `failure_cases.csv`/`self_correction_routes.csv` luôn bằng 0. Nói cách khác, ở quy mô 400 chunk / 174 edge của bài lab, cơ chế `SUPER_NODE_EDGE_CAP = 50` chưa từng được kích hoạt trong thực tế — nó đã được implement và verify đúng logic (`test_supernode_policy()`, cell 5.1) nhưng chưa được stress-test bằng dữ liệu thật có node bậc cao.
- **Ưu điểm (khi triển khai ở quy mô lớn hơn):** Nếu scale lên hàng chục nghìn bài báo, các entity như *OpenAI, Google, Amazon* sẽ dễ dàng vượt degree 100 (xuất hiện trong hàng trăm/nghìn bài). Khi đó, việc ưu tiên 50 cạnh có `published_date` mới nhất (`ORDER BY published_date DESC LIMIT 50`) giữ context nằm trong `MAX_GRAPH_CONTEXT_CHARS = 14000`, tránh bùng nổ BFS và giữ câu trả lời bám sát diễn biến gần nhất — đúng như thiết kế của `retrieve_graph_context()`.
- **Rủi ro:** Cũng chính vì ưu tiên "mới nhất", các câu hỏi cần **toàn bộ chuỗi thời gian** (timeline ordering) sẽ bị ảnh hưởng ngay cả khi chưa chạm ngưỡng super-node — bằng chứng gián tiếp nằm ở `self_correction_routes.csv`: câu `G5000-31` (yêu cầu sắp xếp đúng trình tự các bước OpenAI từ tháng 3 đến tháng 7) phải trải qua **cả 3 vòng self-correction** (`hop2` → `hop3` → `hop3+vector`, tổng 1.493 token, 2.39s) mà vẫn được đánh dấu `sufficient: null` ở vòng cuối — cho thấy việc mở rộng hop không tự động giải quyết được vấn đề thứ tự thời gian; nếu entity như *OpenAI* thực sự trở thành super-node ở quy mô lớn hơn, chính sách cap-theo-độ-mới sẽ càng làm trầm trọng thêm rủi ro cắt mất mốc thời gian cũ cần thiết cho suy luận trình tự.

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

#### Bảng tổng hợp Benchmark (LLM-as-a-Judge, toàn bộ 25 câu — `ALL`) — từ `graphrag_vs_flatrag_summary.csv`:

| Tiêu chí đánh giá | Flat RAG | GraphRAG | Δ (Graph − Flat) | Nhận xét |
|-------------------|----------|----------|-------------------|----------|
| **Comprehensiveness (1–5)** | 3.40 | 4.16 | **+0.76** | GraphRAG cải thiện rõ rệt. |
| **Faithfulness (1–5)** | 3.44 | 4.24 | **+0.80** | GraphRAG cải thiện rõ rệt. |
| **Multi-hop Reasoning (1–5)** | 3.40 | 4.12 | **+0.72** | GraphRAG cải thiện rõ rệt. |
| **Latency trung bình (s)** | 2.362 | 4.662 | **+2.30 (≈ +97%)** | GraphRAG chậm hơn đáng kể — chi phí self-correction (mục 5). |
| **Token usage trung bình** | 737.36 | 855.60 | **+118.24 (≈ +16%)** | Context graph + vòng lặp self-correction tốn thêm token. |

#### Bóc tách theo nhóm câu hỏi:

| Nhóm | Comprehensiveness (Flat→Graph) | Faithfulness (Flat→Graph) | Latency (Flat→Graph) | Nhận xét |
|------|-------------------------------|----------------------------|------------------------|----------|
| **cross-doc** (11 câu) | 3.182 → 4.545 (**+1.364**) | 3.182 → 4.727 (**+1.545**) | 1.927s → 4.596s (**+139%**) | GraphRAG thắng rõ rệt nhất ở nhóm này — đúng như kỳ vọng vì đây là nhóm cần nối thông tin xuyên tài liệu. |
| **multi-hop** (12 câu) | 3.333 → 3.667 (**+0.333**) | 3.417 → 3.667 (**+0.25**) | 2.95s → 4.978s (**+69%**) | Cải thiện khiêm tốn hơn — bị kéo xuống bởi 2 ca lỗi (`G5000-42`, `G5000-30`, xem dưới). |
| **factoid** (2 câu) | 5.0 → 5.0 (**0**) | 5.0 → 5.0 (**0**) | 1.227s → 3.134s (**+156%**) | Chất lượng ngang nhau tuyệt đối (câu hỏi 1-hop không cần graph) nhưng GraphRAG vẫn tốn thêm latency do luôn seed-match + BFS trước khi trả lời. |

**Nhận xét chung:** Ở quy mô 25 câu hỏi này, GraphRAG **thắng rõ về chất lượng**, đặc biệt ở nhóm `cross-doc` (+1.36 → +1.55 điểm/5) — ngược hẳn với trực giác "graph chỉ nhanh hơn nhưng chất lượng ngang nhau" của các lần thử nghiệm với schema hẹp hơn. Cái giá phải trả là **latency gần gấp đôi** và **token cao hơn ~16%**, chủ yếu đến từ cơ chế self-correction (Bonus, mục 5) chứ không phải bản thân retrieval graph.

#### Phân tích 2 Ca lỗi Điển hình:

1. **Ca lỗi Flat RAG thất bại (GraphRAG thành công) — `G5000-33`:**
   - *Câu hỏi:* "Which July OpenAI-related event is a content/technology collaboration, and which July event is a voluntary governance commitment?" (điểm: Flat **1/5** → GraphRAG **5/5**, nhóm `cross-doc`)
   - *Flat RAG:* trả lời đúng vế 1 (thoả thuận AP–OpenAI) nhưng bỏ trắng vế 2: *"The event that represents a voluntary governance commitment is not specified in the provided context"* — vector top-k cho câu hỏi ghép hai sự kiện chỉ kéo về được 1 trong 2 chunk liên quan (`r00366::c0000`), bỏ lỡ `r03330::c0000` (cam kết White House) vì nó không đủ tương đồng embedding với câu hỏi gộp.
   - *GraphRAG:* Seed "OpenAI" khớp trực tiếp trong Neo4j, BFS mở rộng 6 cạnh (`graph_collected_edges=6`) và trả lời đủ cả hai sự kiện: *"the agreement between the Associated Press and OpenAI... [chunk_id=r00366::c0000]"* và *"the pledges made by OpenAI and other tech companies to the White House... [chunk_id=r03330::c0000]"*. Đây chính là ưu thế cốt lõi của graph traversal: nối các sự kiện rời rạc qua cạnh quan hệ tường minh, không phụ thuộc vào độ tương đồng ngữ nghĩa của câu hỏi gộp.

2. **Ca lỗi GraphRAG thất bại — `G5000-30`:**
   - *Câu hỏi:* "Meta appears in two different AI contexts in the selected data. What are they, and what distinct relation should the graph store in each case?" (điểm: Flat **2/5** → GraphRAG **1/5**, nhóm `multi-hop` — GraphRAG kém hơn cả Flat RAG)
   - *Reference answer thật sự:* Meta là nhà cung cấp Llama 2/Code Llama trên Google Cloud Next '23, và Meta là một trong các công ty cam kết AI với Nhà Trắng.
   - *GraphRAG trả lời sai hoàn toàn:* *"Meta is related to the stock market, specifically being listed with NASDAQ under the ticker symbol AAPL"* (đây là ticker của **Apple**, không phải Meta!) và *"Meta's Subscription Service Launch"* — cả hai đều lạc đề so với 2 bối cảnh AI thật sự. Judge nhận xét: *"the claims about Meta's stock listing are incorrect, as AAPL is the ticker for Apple, not Meta."*
   - *Nguyên nhân kỹ thuật (đã truy được tận gốc, mục 2):* node `Product` "NASDAQ: AAPL" bị `merge_guard()` gộp nhầm với "NASDAQ: META" (`MERGE_LEXICAL`, ratio 0.73) dù cosine similarity chỉ 0.659 — tạo ra một node lai khiến khi BFS từ seed "Meta" đi qua cạnh liên kết tới node ticker này, model suy diễn nhầm quan hệ niêm yết. Đây là ví dụ rõ ràng cho thấy **một lỗi Entity Resolution có thể lan trực tiếp thành hallucination ở tầng answer generation**, đúng như rủi ro đã cảnh báo ở mục 1–2.
   - *Đề xuất khắc phục:* (a) thêm sàn tối thiểu cho cosine similarity ngay cả trên nhánh lexical-only (ví dụ ≥ 0.5) để chặn các cặp như AAPL/META lọt qua chỉ vì trùng template chuỗi; (b) xử lý đặc biệt các chuỗi dạng `"NASDAQ: <ticker>"` — so khớp phần sau dấu `:` thay vì so cả chuỗi; (c) bổ sung bước hậu kiểm `context_sufficient()` (đã có ở Bonus self-correction) để phát hiện khi context chứa entity lệch chủ đề câu hỏi trước khi đưa vào answer.

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent

*Trả lời:*
- **Đánh đổi Quality vs Latency vs Token (dữ liệu thật, mục 4):** GraphRAG đạt chất lượng cao hơn rõ rệt (+0.72 → +0.80 điểm/5 trung bình, tới +1.55 ở `cross-doc`) nhưng **latency tăng ~97%** và **token tăng ~16%**. Truy vết bằng `outputs/self_correction_routes.csv` cho thấy nguyên nhân không nằm ở bản thân BFS/textualize mà ở **vòng lặp self-correction** (Bonus, `self_correcting_context()`): với các câu multi-hop/cross-doc khó (ví dụ `G5000-27`, `G5000-30`, `G5000-31`, `G5000-32`, `G5000-37`, `G5000-38`, `G5000-39`, `G5000-42`, `G5000-48`, `G5000-49`), route `hop2` ban đầu bị `context_sufficient()` đánh giá không đủ, hệ thống tự động thử `hop3` rồi `hop3+vector` — mỗi vòng cộng thêm ít nhất một lệnh gọi LLM kiểm tra tính đầy đủ + một lượt truy vấn mở rộng, khiến tổng token cho các câu này vọt lên 1.000–1.500 token (so với 300–800 token của các câu chỉ cần `hop2`). Đây là chi phí xác đáng: chính cơ chế self-correction là thứ đã "mua" được phần lớn khoản cải thiện +0.76→0.80 điểm chất lượng nói trên, chứ không phải retrieval graph đơn thuần.
- **Quyết định từ chối đề xuất của AI Coding Agent:** Ở "AI Coding Agent Challenge A — Near Dedup" (cell 1.5/1.9), đề bài quy định rõ **không chấp nhận** giải pháp so khớp cosine similarity theo cặp `O(N²)` trên toàn corpus. Kết quả thực tế: `outputs/near_dedup_audit.csv` áp dụng ước lượng Jaccard kiểu MinHash/LSH trên 440 cặp ứng viên (không phải toàn bộ N² cặp của 2.118 bài), phát hiện **18 cặp** vượt ngưỡng 0.85 (`NEAR_DUP_MERGE`) — ví dụ `r00033`/`r01746` ("Aeris to Acquire IoT Business from Ericsson", jaccard≈0.969) hay `r00891`/`r03296` (bài L&T–Qualcomm–Thales, jaccard≈0.859). Chính sách áp dụng là **`audit`** (chỉ ghi nhận, không tự động xoá) vì hai câu hỏi Golden Dataset `G5000-36` và `G5000-45` **chủ động yêu cầu cả hai bản gần trùng lặp phải còn tồn tại trong corpus** để kiểm tra khả năng canonicalize-không-xoá của graph (tránh double-counting bằng cách hợp nhất sự kiện, không phải bằng cách xoá dữ liệu).
- **Giải pháp scale lên ~350MB (~100.000 bài báo):** Với chi phí LLM đo được ở quy mô 400 chunk (extraction: 181 call / 183.152 token; answer: 82 call / 45.111 token; judge: 52 call / 53.536 token — theo `outputs/report_facts.md` mục 8), bottleneck đầu tiên chắc chắn là **NER/RE qua LLM** (`run_extraction`, batch tuần tự `batch_size=4`) — ở quy mô 100k bài sẽ cần hàng trăm nghìn lệnh gọi LLM. Giải pháp: (1) song song hoá bằng async batch/worker queue có rate-limit throttling; (2) chuyển FAISS `IndexFlatIP` sang `IndexHNSWFlat` (ANN) cho Entity Resolution để tránh so khớp toàn cục khi số thực thể tăng; (3) tận dụng module Bonus **Community Detection** đã có sẵn (`build_communities()`, `networkx.algorithms.community.greedy_modularity_communities`, xem `outputs/community_reports.csv`) để phân vùng đồ thị theo cộng đồng, giúp BFS/retrieval không phải quét toàn graph và giúp super-node mitigation hoạt động theo từng cộng đồng cục bộ thay vì toàn cục; (4) vì self-correction đã cho thấy chi phí 2–3 lần LLM round-trip trên các câu khó (mục này), cần giới hạn số vòng self-correction hoặc cache kết quả `context_sufficient()` theo cặp (câu hỏi, subgraph) để chi phí không nhân lên tuyến tính theo lưu lượng truy vấn ở production.

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN (Reflection & Action Plan)

### 1. Mapping Bài giảng vào Code

| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|--------------------------|------------------|------------------------|-----------------------------|
| **Conservative Coreference** | Module 1 (cell 1.7) | `resolve_coref_batch()` | 0/400 chunk có `unresolved_mentions`, không phát sinh False Edge do coref trong lần chạy này — nhưng rủi ro đồng xuất hiện dày đặc (mục 1) vẫn tồn tại về mặt cấu trúc dữ liệu. |
| **Schema & Allowlist Guard** | Module 2 (cell 2.1) | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` (CORE 8 quan hệ theo ASSIGNMENT) | Chỉ **48,6%** triple trích xuất nằm trong 8 quan hệ CORE — phần còn lại cần schema mở rộng (27 quan hệ / 5 loại node, `outputs/report_facts.md` mục 2) vì chính Golden Dataset yêu cầu các quan hệ như `PROVIDES_MODEL`, `COMMITTED_TO` (ví dụ `G5000-30`) nằm ngoài 8 quan hệ CORE. |
| **Entity Resolution & Union-Find** | Module 3 (cell 2.2) | `build_resolution_map()`, `merge_guard()`, `UF` | Chặn đúng cặp có ngữ nghĩa gần nhưng khác cấp (`Middlebury Institute` vs `Middlebury Institute community`, sim 0.925) — nhưng để lọt một False Merge nguy hiểm (`NASDAQ: AAPL` ↔ `NASDAQ: META`, ratio 0.73) chính là nguyên nhân gốc của ca lỗi `G5000-30`. |
| **Bulk Cypher Ingestion** | Module 2 (cell 2.3–2.4) | `bulk_insert_nodes()`, `bulk_insert_edges()`, `graph_checks()` | `UNWIND` batch 1000; `graph_checks()` xác nhận **0 cạnh thiếu provenance** trên 174 edge / 287 node. |
| **Super-node Degree Cap** | Module 4 (cell 3.3, 5.1) | `retrieve_graph_context()`, `test_supernode_policy()` | Logic cap đúng thiết kế nhưng **chưa từng được kích hoạt thực tế** ở quy mô lab (degree cao nhất = 6 ≪ ngưỡng 100) — cần dữ liệu lớn hơn để kiểm chứng thực nghiệm. |
| **LLM-as-a-Judge Evaluation** | Module 5 (cell 4.2) | `judge_answer()` | Chấm nhất quán trên 25/25 câu, phát hiện rõ khoảng cách chất lượng lớn (tới +1.5 điểm ở `cross-doc`) — đủ nhạy để phân biệt hai hệ thống thay vì chỉ đánh giá định tính. |
| **(Bonus) Self-Correction** | Bonus | `self_correcting_context()`, `context_sufficient()` | Là nguyên nhân chính khiến latency GraphRAG tăng ~97% — đổi lại giúp các câu multi-hop/cross-doc khó có cơ hội mở rộng ngữ cảnh qua 2–3 vòng thay vì trả lời thiếu ngay từ `hop2`. |
| **(Bonus) Community Detection** | Bonus | `build_communities()` (NetworkX `greedy_modularity_communities`) | Phát hiện 5 cộng đồng mạch lạc (`outputs/community_reports.csv`): White House/AI policy, ServiceNow GenAI, L&T partnerships, Amazon AI/climate, Academic-NASDAQ cluster — tiềm năng dùng làm đơn vị phân vùng retrieval khi scale lớn (mục 5). |

---

### 2. Quá trình Debugging & Bài học
- **Lỗi kỹ thuật phức tạp nhất gặp phải:** False Merge `NASDAQ: AAPL` ↔ `NASDAQ: META` trong Entity Resolution (mục 2) — khó phát hiện vì cosine similarity của cặp này (0.659) **thấp hơn hẳn** ngưỡng vector 0.90, nên trực giác ban đầu là "cặp này chắc chắn bị `REJECT_BELOW_THRESHOLD`" — nhưng nó vẫn lọt qua vì đạt `MERGE_LEXICAL` qua nhánh so khớp chuỗi thuần túy (`SequenceMatcher` ratio 0.73 ≥ 0.72), một nhánh không bị chặn thêm bởi sàn similarity tối thiểu. Lỗi chỉ lộ ra khi LLM Judge chấm điểm thấp bất thường cho `G5000-30` và rationale nêu rõ *"AAPL is the ticker for Apple, not Meta."*
- **Cách xử lý và bài học:** Thay vì debug ngược từ prompt trả lời (`answer_prompt`), tôi tra ngược **từ điểm Judge thấp bất thường → `graphrag_eval_results.csv` (câu trả lời sai cụ thể) → `entity_resolution_audit.csv` (lọc theo tên thực thể xuất hiện trong câu trả lời sai)** để tìm đúng dòng gây lỗi — cách tiếp cận "trace từ output ngược về audit log" này hiệu quả hơn nhiều so với đọc lại toàn bộ prompt trích xuất. Bài học rút ra: **Lexical Guard không đồng nhất về độ an toàn giữa chuỗi dài và chuỗi ngắn có template lặp** — chuỗi càng ngắn/càng có cấu trúc mẫu cố định (như `"NASDAQ: X"`) thì `SequenceMatcher.ratio()` càng dễ cho điểm cao giả tạo bất kể nội dung khác biệt, nên cần một sàn similarity riêng (hoặc luật đặc thù theo pattern) cho các trường hợp này thay vì dùng chung một ngưỡng ratio cho mọi độ dài chuỗi.

---

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)
- **Tên đồ án / Dự án:** Tech Partnership & Investment Intelligence Graph — hệ thống theo dõi quan hệ đối tác/đầu tư/công nghệ giữa các công ty công nghệ theo thời gian.
- **Đặc thù bài toán & Lý do chọn giải pháp:** Dữ liệu tin tức công nghệ (M&A, đầu tư, hợp tác) có bản chất đa chặng, xuyên tài liệu và thay đổi theo thời gian — đúng kịch bản mà thực nghiệm mục 4 cho thấy GraphRAG vượt trội rõ nhất (nhóm `cross-doc`: +1.36 → +1.55 điểm/5). Tuy nhiên, bài học từ `G5000-30` cũng cho thấy cần đầu tư nghiêm túc vào Entity Resolution ngay từ đầu, không chỉ vào retrieval.
- **Cấu trúc Node & Relation dự kiến:** Rút kinh nghiệm từ việc CORE 8 quan hệ chỉ phủ được 48,6% triple thực tế của Lab 19, đồ án sẽ **thiết kế schema mở rộng được (extensible) ngay từ đầu** thay vì cố định cứng 8 quan hệ:
  - Nodes: `Company`, `Person`, `Technology/Product`, `Event` (tách riêng Event để lưu mốc thời gian, giảm rủi ro nhầm lẫn thời điểm như case AMD/AWS "tentative" ở `G5000-27`).
  - Relations lõi: `ACQUIRED`, `INVESTED_IN`, `PARTNERED_WITH`; relations mở rộng theo nhu cầu thực tế: `PROVIDES_MODEL`, `COMMITTED_TO`, `EVALUATING` (tách biệt với `PARTNERED_WITH` đã xác nhận, tránh nhầm "đang cân nhắc" với "đã hợp tác" như bài học AWS–AMD).
- **Chiến lược xử lý Entity Resolution & Super-node:** Giữ cơ chế 2 lớp (vector threshold 0.90 + lexical guard ratio 0.72) vì đã chứng minh hiệu quả với chuỗi tên dài (case Middlebury), **nhưng bổ sung thêm sàn cosine similarity tối thiểu cho nhánh lexical-only** và luật riêng cho các chuỗi có template ngắn/lặp (ticker, mã sản phẩm) để tránh lặp lại lỗi `NASDAQ: AAPL`/`NASDAQ: META`. Với Super-node, áp dụng cap theo độ mới như Lab 19 nhưng cân nhắc thêm **cap theo loại quan hệ** (giữ đủ lịch sử cho `ACQUIRED`/`INVESTED_IN` vì hiếm xảy ra, chỉ cap mạnh các quan hệ tần suất cao) — đồng thời cân nhắc dùng module Community Detection (đã kiểm chứng ở Bonus Lab 19) để phân vùng retrieval khi đồ thị đủ lớn để super-node cap thực sự bị kích hoạt.

---

## 🎯 TỰ ĐÁNH GIÁ
| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|----------|-------------------|---------|
| Mức độ hiểu bài giảng GraphRAG | 4 | Nắm được luồng coreference → NER/RE → entity resolution → graph retrieval → self-correction, và giải thích được vì sao chi phí latency tăng (không phải do BFS mà do vòng lặp self-correction). |
| Khả năng kiểm soát AI Coding Agent | 4 | Từ chối đúng giải pháp `O(N²)` cho near-dedup; xác nhận được bằng số liệu thật (440 cặp ứng viên, 18 cặp merge) rằng chính sách `audit`-only là lựa chọn đúng cho yêu cầu golden set. |
| Chất lượng đồ thị tri thức xây dựng | 3 | 0 cạnh thiếu provenance và Guard chặn đúng phần lớn trường hợp phức tạp, nhưng vẫn để lọt một False Merge (`NASDAQ: AAPL`/`META`) gây hallucination trực tiếp ở `G5000-30` — cho thấy guard chưa robust với chuỗi ngắn có template. |
| Khả năng phân tích và debug hệ thống | 5 | Truy ngược thành công từ điểm Judge bất thường ở `G5000-30` về đúng dòng audit gây lỗi trong `entity_resolution_audit.csv`, xác định chính xác cơ chế (`MERGE_LEXICAL` qua nhánh không gated bởi sàn similarity) thay vì chỉ dừng ở mức "GraphRAG trả lời sai". |
