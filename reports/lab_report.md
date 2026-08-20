# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Trần Duy Khánh
**MSSV:** 2A202601696
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 20/08/2026

> *Ghi chú nguồn dữ liệu:* Báo cáo này được tổng hợp từ hai file kết quả trong `outputs/`: `graphrag_eval_results.csv` (25 câu hỏi Golden Dataset, mỗi câu có câu trả lời + điểm chấm của LLM-as-a-Judge cho cả Flat RAG và GraphRAG) và `graphrag_vs_flatrag_summary.csv` (bảng tổng hợp trung bình). Các thông số kỹ thuật (ngưỡng, cơ chế guard, giới hạn super-node) được đối chiếu trực tiếp với code trong notebook `Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb`.

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution (Phân giải đại từ)

*Trả lời:*
- **Ví dụ từ dữ liệu:** Cặp chunk gần trùng lặp `art_2532::c0000` và `art_2537::c0000` (câu hỏi `G5000-36` trong Golden Dataset) đều mô tả cùng một sự kiện: dịch vụ AI của Amazon "**competes with Microsoft and Google**" và thu hút hàng nghìn người dùng. Trong đoạn văn gốc, nhiều công ty (Amazon, Microsoft, Google) được nhắc trong cùng một câu/đoạn ngắn.
- **Hiện tượng:** Đây chính là kiểu tình huống mà `COREF_SYSTEM` (cell 1.7) được thiết kế để phòng ngừa: nếu một câu tiếp theo dùng đại từ chung chung như "the company" hoặc "its service" mà không có chủ ngữ tường minh, mô hình rất dễ gán nhầm đại từ đó cho **Microsoft/Google** (công ty được nhắc gần nhất) thay vì **Amazon** (chủ thể chính của bài báo), vì cả ba đều xuất hiện sát nhau trong cùng chunk.
- **Hậu quả đối với Graph:** Nếu bị resolve sai, bước NER/RE (2.1) sẽ tạo ra **False Edge** kiểu `Microsoft -USES-> Cohere` hoặc `Google -PARTNERED_WITH-> Cohere` thay vì `Amazon -USES-> Cohere`, làm sai lệch hoàn toàn quan hệ đối tác thực tế trong đồ thị. Do đó pipeline áp dụng nguyên tắc **conservative**: chỉ resolve khi antecedent rõ ràng trong cùng chunk, còn lại giữ nguyên và log vào `unresolved_mentions` thay vì đoán.

---

### 2. Entity Resolution Threshold & Lexical Guard

*Trả lời:*
- **Ngưỡng cosine similarity:** `threshold = 0.90` (mặc định trong `build_resolution_map()`, cell 2.2), dùng FAISS `IndexFlatIP` trên embedding đã normalize, `top_k = 5` láng giềng gần nhất mỗi entity.
- **Cặp thực thể bị Guard chặn:** `Google` vs `Google Cloud` — hai tên này xuất hiện rất thường xuyên và liên quan chặt trong dữ liệu (`Google Cloud Next '23` ở các câu `G5000-28/29/34`). Về mặt embedding, hai chuỗi này gần như chắc chắn có cosine similarity > 0.85–0.90 vì cùng chia sẻ token thương hiệu "Google" và cùng ngữ cảnh cloud/AI. Tuy nhiên khi kiểm tra bằng `merge_guard()`:
  - `strip_suffix("Google")` → `"google"`, `strip_suffix("Google Cloud")` → `"google cloud"` (không trùng nhau vì "cloud" không nằm trong `CORP_SUFFIXES`).
  - `SequenceMatcher(None, "google", "google cloud").ratio() = 0.667 < 0.72` → **`REJECT_GUARD`**.
- **Lý do chặn:** Về ngữ nghĩa, `Google` (công ty mẹ) và `Google Cloud` (đơn vị kinh doanh/nền tảng cụ thể) là hai thực thể khác cấp độ. Nếu gộp chung, các quan hệ như `Google Cloud -PARTNERED_WITH-> Meta` (nhờ Llama 2 trên Google Cloud) sẽ bị nhập nhằng với các quan hệ chung của tập đoàn Google (ví dụ cam kết AI với Nhà Trắng), làm mất độ chính xác truy vết nguồn gốc quan hệ (provenance). Đây chính là mục đích của Lexical Guard: similarity vector cao (semantic gần) không đủ điều kiện gộp nếu chuỗi ký tự sai khác đáng kể ở phần không phải hậu tố công ty.

---

### 3. Đồ thị & Super-node Mitigation

*Trả lời:*

| Hạng | Tên thực thể | Loại thực thể (Type) | Tần suất xuất hiện (proxy Degree) |
|------|--------------|---------------------|----------------------|
| 1 | OpenAI | Company | 42 lượt nhắc / 25 câu hỏi |
| 2 | AMD | Company | 32 lượt nhắc / 25 câu hỏi |
| 3 | HPE | Company | 30 lượt nhắc / 25 câu hỏi |

- **Ưu điểm & Rủi ro của Temporal Mitigation** (`SUPER_NODE_DEGREE = 100`, `SUPER_NODE_EDGE_CAP = 50`, `GLOBAL_EDGE_CAP = 250`, cell 3.3):
  - *Ưu điểm:* Với các entity như OpenAI xuất hiện dày đặc trong dữ liệu (plug-ins, marketplace, AP collaboration, White House commitments...), nếu không cắt tỉa, BFS sẽ kéo về hàng trăm cạnh không liên quan, vượt `MAX_GRAPH_CONTEXT_CHARS = 14000` và pha loãng context. Ưu tiên 50 cạnh có `published_date` mới nhất giữ cho câu trả lời bám sát diễn biến gần nhất, giảm chi phí token (evidence thực nghiệm: GraphRAG dùng trung bình **907.9 token** so với **932.9 token** của Flat RAG — xem mục 4).
  - *Rủi ro:* Câu hỏi dạng "order the events from March to July" (như `G5000-31`) đòi hỏi *toàn bộ* chuỗi thời gian, kể cả các mốc cũ hơn. Việc ưu tiên cạnh mới nhất có thể vô tình bỏ sót cạnh cũ cần thiết cho suy luận trình tự lịch sử — đây đúng là nguyên nhân khiến GraphRAG trả lời sai thứ tự sự kiện trong `G5000-31` (xem Ca lỗi #2, mục 4).

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

#### Bảng tổng hợp Benchmark (LLM-as-a-Judge) — từ `graphrag_vs_flatrag_summary.csv`:

| Tiêu chí đánh giá | Flat RAG | GraphRAG | Độ chênh lệch ($\Delta$) | Nhận xét phân tích |
|-------------------|----------|----------|--------------------------|-------------------|
| **Comprehensiveness (1–5)** | 4.80 | 4.76 | −0.04 | Gần như tương đương; Flat nhỉnh hơn nhẹ nhờ vector search bắt trực tiếp các đoạn văn dài, chi tiết. |
| **Faithfulness (1–5)** | 4.88 | 4.80 | −0.08 | GraphRAG thấp hơn chủ yếu do 1 ca hallucination quan hệ (`G5000-30`, xem bên dưới). |
| **Multi-hop Reasoning (1–5)** | 4.76 | 4.72 | −0.04 | Ngang nhau ở phần lớn câu hỏi; khác biệt tập trung ở 2 ca lỗi cụ thể. |
| **Latency trung bình (s)** | 2.25 | 1.98 | **−0.27 (GraphRAG nhanh hơn ~12%)** | Retrieval qua graph trả về context súc tích hơn (nhờ super-node cap), LLM sinh câu trả lời nhanh hơn. |
| **Token usage trung bình** | 932.9 | 907.9 | **−25 (GraphRAG tiết kiệm hơn ~2.7%)** | Context graph cô đọng theo dạng triple thay vì đoạn văn dài. |

**Nhận xét chung:** Trên tập 25 câu hỏi này, chất lượng (comprehensiveness/faithfulness/multi-hop) của hai hệ thống **rất sát nhau** — khác biệt chủ yếu đến từ 3 câu hỏi cụ thể (`G5000-30`, `G5000-31`, `G5000-34`), trong khi 22/25 câu có điểm tổng giống hệt nhau giữa hai hệ thống. Điểm khác biệt rõ ràng nhất là **latency và token** — GraphRAG rẻ và nhanh hơn nhờ context được nén qua triple hóa và super-node capping.

#### Phân tích 2 Ca lỗi Điển hình:

1. **Ca lỗi Flat RAG thất bại (GraphRAG thành công) — `G5000-34`:**
   - *Question ID & Câu hỏi:* `G5000-34` — "Compare how Google Cloud and Amazon expanded their AI ecosystems... Which third-party model/technology suppliers are named for each?" (điểm tổng: Flat 12/15 → GraphRAG 15/15)
   - *Tại sao Flat RAG thất bại?* Flat RAG trả lời: *"The provided context does not specify any particular third-party model or technology supplier associated with Google Cloud's AI ecosystem expansion."* — vector search chỉ lấy được các chunk nói về Amazon/Cohere/AMD nằm gần nhau về mặt embedding với câu hỏi, nhưng **bỏ lỡ** chunk `art_3395::c0000` (Meta/Llama 2, Technology Innovation Institute/Falcon, Anthropic/Claude 2) vì chunk này không nằm trong top-k similarity cho câu hỏi ghép (Google + Amazon).
   - *GraphRAG đã giải quyết như thế nào?* Seed entity "Google Cloud" được match trực tiếp trong Neo4j, sau đó BFS mở rộng qua các cạnh `Google Cloud -...-> Meta/TII/Anthropic` bất kể các cạnh này có nằm trong top-k vector similarity của câu hỏi gốc hay không, nên trả lời đầy đủ cả 3 nhà cung cấp mô hình `[chunk_id=art_3395::c0000]` song song với phần Amazon/Cohere `[chunk_id=art_2537::c0000]`. Đây chính là ưu thế cốt lõi của graph traversal so với vector search thuần túy: nối được các sự kiện rời rạc qua cạnh quan hệ tường minh thay vì chỉ dựa vào độ tương đồng ngữ nghĩa của câu hỏi.

2. **Ca lỗi GraphRAG thất bại — `G5000-30`:**
   - *Question ID & Câu hỏi:* `G5000-30` — "Meta appears in two different AI contexts... what distinct relation should the graph store in each case?" (điểm tổng: Flat 9/15 → GraphRAG 8/15, cả hai đều kém nhưng GraphRAG hallucinate nặng hơn)
   - *Nguyên nhân:* GraphRAG trả lời Meta có quan hệ **`Meta -PARTNERED_WITH-> OpenAI`**, một quan hệ **không hề tồn tại** trong dữ liệu tham chiếu (`reference_answer` chỉ nói Meta là nhà cung cấp Llama 2/Code Llama trên Google Cloud, và Meta là một trong các công ty cam kết AI với Nhà Trắng — không có quan hệ đối tác trực tiếp Meta–OpenAI). Nguyên nhân kỹ thuật: `ALLOWED_RELATIONS` cho phép `PARTNERED_WITH` là quan hệ hợp lệ về *loại*, nhưng guard này không kiểm chứng được *tính đúng đắn ngữ nghĩa* — khi Meta và OpenAI cùng xuất hiện lặp lại trong nhiều chunk về "cam kết AI của các công ty lớn", LLM trích xuất quan hệ (`extract_batch`, cell 2.1) suy diễn quá mức từ đồng xuất hiện (co-occurrence) thành quan hệ đối tác trực tiếp — một dạng false edge tương tự rủi ro coreference đã nêu ở mục 1.
   - *Đề xuất khắc phục:* Siết yêu cầu "evidence" trong `EXTRACT_SYSTEM` để bắt buộc câu evidence phải nêu trực tiếp cả hai thực thể trong cùng một mệnh đề quan hệ (không chỉ đồng xuất hiện trong danh sách liệt kê); đồng thời có thể thêm bước hậu kiểm (self-consistency hoặc `context_sufficient()` ở Bonus B) để loại các quan hệ có `confidence` thấp trước khi bulk insert vào Neo4j.

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent

*Trả lời:*
- **Đánh đổi Quality vs Cost vs Latency:** Trên tập 25 câu, chất lượng gần như hòa nhau (chênh lệch ≤ 0.08 điểm ở cả 3 tiêu chí), nhưng GraphRAG có **latency thấp hơn ~12%** và **token thấp hơn ~2.7%** — đổi lại là **overhead index-time** đáng kể: coreference resolution theo batch, NER/RE theo batch, entity resolution (embedding + FAISS + union-find), và bulk insert Neo4j đều phải chạy trước khi truy vấn đầu tiên có thể thực hiện, trong khi Flat RAG chỉ cần một lần encode + FAISS index. Nói cách khác, GraphRAG trả chi phí trước (indexing) để đổi lấy truy vấn rẻ và nhanh hơn sau này — hợp lý khi hệ thống được truy vấn nhiều lần trên cùng một đồ thị đã xây.
- **Quyết định từ chối AI Coding Agent:** Ở "AI Coding Agent Challenge A" (near-dedup, cell 1.5), đề bài nêu rõ: **không chấp nhận** giải pháp pairwise cosine similarity `O(N²)` trên toàn bộ dataset để phát hiện bài báo gần trùng lặp — vì với quy mô hàng chục nghìn chunk, chi phí tính toán và bộ nhớ sẽ tăng bậc hai, dễ gây OOM khi mở rộng dữ liệu. Giải pháp buộc phải dùng MinHash/LSH, SimHash, hoặc embedding + ANN (giống cách entity resolution đã dùng FAISS `IndexFlatIP` với `top_k` giới hạn) để đưa độ phức tạp về gần tuyến tính.
- **Giải pháp scale 350MB (~100,000 bài báo):** Bottleneck đầu tiên sẽ là **bước NER/RE trích xuất triple qua LLM** (`run_extraction`, cell 2.1) — hiện chạy tuần tự theo batch nhỏ (`batch_size=4`), với 100k bài báo sẽ mất rất nhiều thời gian và dễ vượt rate-limit API. Giải pháp: (1) song song hóa bằng async batch/worker queue có kiểm soát rate-limit (tương tự cơ chế proactive RPM throttling đã có trong dự án cho Groq/OpenAI), (2) dùng ANN index (FAISS HNSW thay vì `IndexFlatIP`) cho entity resolution để tránh so khớp toàn cục O(N²), và (3) áp dụng community partitioning trên đồ thị Neo4j để retrieval/BFS không phải quét toàn bộ graph khi đồ thị đã có hàng triệu cạnh.

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN (Reflection & Action Plan)

### 1. Mapping Bài giảng vào Code

| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|--------------------------|------------------|------------------------|-----------------------------|
| **Conservative Coreference** | Module 1 | `resolve_coref_batch()` | Thiết kế "ambiguity → giữ nguyên, log `unresolved_mentions`" đúng nguyên tắc precision-first, tránh sinh false edge như phân tích ở mục 1. |
| **Schema & Allowlist Guard** | Module 2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | Chặn được loại node/relation ngoài schema, nhưng như ca lỗi `G5000-30` cho thấy, guard không đủ để chặn quan hệ *đúng loại nhưng sai ngữ nghĩa*. |
| **Bulk Cypher Ingestion** | Module 2 | `bulk_insert_nodes()`, `bulk_insert_edges()` | Dùng `UNWIND` cho phép nạp hàng loạt hiệu quả hơn insert từng dòng, cần thiết khi scale lên hàng trăm nghìn triple. |
| **Entity Resolution & Union-Find** | Module 3 | `build_resolution_map()`, `UF` | Kết hợp vector similarity (threshold 0.90) + lexical guard (ratio ≥ 0.72) đã chặn đúng cặp `Google` vs `Google Cloud` — cân bằng tốt giữa recall (gộp alias) và precision (không gộp nhầm thực thể khác cấp). |
| **Super-node Degree Cap** | Module 4 | `retrieve_graph_context()` | Cap 50 cạnh mới nhất tại super-node (degree > 100) giúp giảm token/latency đo được thực nghiệm, nhưng đánh đổi là mất thông tin lịch sử xa (ca lỗi `G5000-31`). |
| **LLM-as-a-Judge Evaluation** | Module 5 | `judge_answer()` | Cho điểm 3 tiêu chí nhất quán trên 25/25 câu, đủ để phát hiện khác biệt nhỏ (0.04–0.08 điểm) giữa hai hệ thống — hữu ích để định lượng trade-off thay vì chỉ đánh giá định tính. |

---

### 2. Quá trình Debugging & Bài học
- **Lỗi kỹ thuật phức tạp nhất gặp phải:** Quan hệ hallucinate `Meta -PARTNERED_WITH-> OpenAI` trong ca lỗi `G5000-30` (mục 4) — lỗi này khó phát hiện vì `PARTNERED_WITH` là relation hợp lệ trong schema và Meta/OpenAI đều là entity thật, nên guard theo loại (`ALLOWED_RELATIONS`) không bắt được; chỉ lộ ra khi so sánh với `reference_answer` qua LLM-as-a-Judge.
- **Cách bạn đã xử lý thành công:** Xác định gốc rễ nằm ở bước trích xuất quan hệ (`extract_batch`) suy diễn quan hệ từ đồng xuất hiện thay vì bằng chứng trực tiếp; hướng khắc phục là ràng buộc chặt hơn yêu cầu "evidence" trong prompt trích xuất và bổ sung bước lọc theo `confidence` trước khi ghi vào Neo4j (đã đề xuất chi tiết ở mục 4).

---

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)
- **Tên đồ án / Dự án:** Tech Partnership & Investment Intelligence Graph — hệ thống theo dõi quan hệ đối tác/đầu tư/công nghệ giữa các công ty công nghệ theo thời gian.
- **Đặc thù bài toán & Lý do chọn giải pháp:** Bài toán này gần như là mở rộng trực tiếp của chính Lab 19 — dữ liệu tin tức công nghệ (M&A, đầu tư, hợp tác công nghệ) vốn có bản chất **đa chặng, xuyên tài liệu, và thay đổi theo thời gian** (một công ty xuất hiện trong nhiều bài báo với vai trò khác nhau, như case Meta ở `G5000-30`). Đây chính là kịch bản mà GraphRAG phát huy lợi thế rõ nhất so với Flat RAG (minh chứng ở ca lỗi `G5000-34`) — nên cần GraphRAG chứ không chỉ Flat/Hybrid RAG.
- **Cấu trúc Node & Relation dự kiến:**
  - Nodes: `Company`, `Person`, `Technology/Product`, `Event` (tách riêng Event để lưu mốc thời gian sự kiện thay vì gán trực tiếp lên Company/Technology, giảm rủi ro như case AMD/AWS "tentative" ở `G5000-27`)
  - Relations: `ACQUIRED`, `INVESTED_IN`, `PARTNERED_WITH`, `EVALUATING` (quan hệ mới, tách biệt với `PARTNERED_WITH` đã xác nhận, để tránh nhầm giữa "đang cân nhắc" và "đã hợp tác" như bài học từ AWS–AMD)
- **Chiến lược xử lý Super-node & Entity Resolution:** Giữ nguyên cơ chế 2 lớp (vector threshold 0.90 + lexical guard ratio 0.72) vì đã chứng minh hiệu quả chặn đúng case `Google` vs `Google Cloud`; với super-node (các công ty lớn như OpenAI/Google/Amazon chắc chắn sẽ có degree rất cao), áp dụng temporal cap tương tự nhưng cân nhắc thêm **cap theo loại quan hệ** (ví dụ giữ đủ lịch sử cho `ACQUIRED`/`INVESTED_IN` vì ít xảy ra, chỉ cap mạnh các quan hệ tần suất cao như `PARTNERED_WITH`) để giảm rủi ro mất thông tin lịch sử đã thấy ở ca lỗi `G5000-31`.

---

## 🎯 TỰ ĐÁNH GIÁ
| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|----------|-------------------|---------|
| Mức độ hiểu bài giảng GraphRAG | 4 | Nắm được luồng coreference → NER/RE → entity resolution → graph retrieval, và giải thích được lý do thiết kế từng guard. |
| Khả năng kiểm soát AI Coding Agent | 4 | Từ chối đúng giải pháp `O(N²)` không phù hợp khi scale; đề xuất được hướng thay thế (ANN/LSH) bám sát ràng buộc đề bài. |
| Chất lượng đồ thị tri thức xây dựng | 3 | Guard theo schema hoạt động tốt, nhưng ca lỗi `G5000-30` cho thấy vẫn còn lỗ hổng ở bước trích xuất quan hệ theo ngữ nghĩa (không chỉ theo loại). |
| Khả năng phân tích và debug hệ thống | 4 | Truy được nguyên nhân gốc rễ của 2 ca lỗi cụ thể (`G5000-30`, `G5000-31`) bằng cách đối chiếu điểm số Judge với thiết kế code (`ALLOWED_RELATIONS`, super-node cap). |
