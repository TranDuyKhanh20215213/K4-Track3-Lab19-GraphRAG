# Plan: Near-Dedup Implementation (Bonus Challenge A, +3 điểm)

**Nguồn:** `ASSIGNMENT.md` / `RUBRIC.md` bonus "Near-Dedup Implementation" — bổ sung lọc trùng lặp
gần (repost/near-duplicate) sau bước exact-hash dedup ở notebook, dùng MinHash/LSH thay vì
pairwise cosine O(N²).

**Phương pháp đã chọn:** MinHash + LSH (word-shingle, k=5) qua thư viện `datasketch`, gom cụm bằng
`networkx` (đã có sẵn trong pipeline) thay vì tự viết Union-Find, để giữ code ngắn gọn và nhất quán
với các dependency đã cài.

**Điểm chèn:** Notebook `Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb`, giữa cell markdown
"🎯 AI Coding Agent Challenge A — Near Dedup" (id `469ca454`) và cell code "1.6 — LLM wrapper" (id
`863fddb0`). Cell "1.5 — Loader + exact dedup + chunking" (id `adb244b9`) chứa `standardize_news`,
`build_chunks` và dòng gọi mẫu bị comment ở cuối — dòng gọi `near_dedup` sẽ được chèn vào đó.

## Task 1 — Thêm dependency `datasketch`
- Sửa `requirements.txt`: thêm dòng `datasketch>=1.6.0`.
- Sửa cell 1.1 trong notebook (id `78d4bd90`): thêm `datasketch` vào chuỗi `%pip -q install ...`.
- Verify: `grep datasketch requirements.txt` và grep trong notebook đều có kết quả.

## Task 2 — Cell code near-dedup mới (chèn sau cell `469ca454`)
Tiêu đề cell: `#@title 1.5b — Near-Dedup (MinHash/LSH) — AI Coding Agent Challenge A`

Nội dung:
```python
from datasketch import MinHash, MinHashLSH
import networkx as nx

NEAR_DEDUP_THRESHOLD = 0.8
NEAR_DEDUP_NUM_PERM = 128
NEAR_DEDUP_SHINGLE_K = 5

def word_shingles(text, k=NEAR_DEDUP_SHINGLE_K):
    words = norm_space(text).lower().split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}

def build_minhash(text, num_perm=NEAR_DEDUP_NUM_PERM):
    mh = MinHash(num_perm=num_perm)
    for sh in word_shingles(text):
        mh.update(sh.encode("utf-8"))
    return mh

def near_dedup(df, threshold=NEAR_DEDUP_THRESHOLD, num_perm=NEAR_DEDUP_NUM_PERM):
    """MinHash/LSH near-duplicate filter. O(N) LSH query per row, not O(N^2) pairwise."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}
    audit_rows = []
    graph = nx.Graph()
    graph.add_nodes_from(df["article_id"].tolist())

    for r in tqdm(df.itertuples(index=False), total=len(df), desc="Near-dedup"):
        mh = build_minhash(f"{r.title}\n{r.text}")
        for cand_id in lsh.query(mh):
            jac = float(mh.jaccard(minhashes[cand_id]))
            decision = "MERGE_NEAR_DUP" if jac >= threshold else "REJECT_LOW_JACCARD"
            audit_rows.append({
                "article_id_a": cand_id, "article_id_b": r.article_id,
                "jaccard": jac, "decision": decision,
            })
            if decision == "MERGE_NEAR_DUP":
                graph.add_edge(cand_id, r.article_id)
        lsh.insert(r.article_id, mh)
        minhashes[r.article_id] = mh

    text_len = df.set_index("article_id")["text"].str.len().to_dict()
    keep_ids, drop_ids = set(), set()
    for cluster in nx.connected_components(graph):
        if len(cluster) == 1:
            keep_ids |= cluster
            continue
        best = max(cluster, key=lambda aid: text_len.get(aid, 0))
        keep_ids.add(best)
        drop_ids |= (cluster - {best})

    before = len(df)
    deduped = df[~df["article_id"].isin(drop_ids)].reset_index(drop=True)
    print(f"Near-dedup: {before:,} -> {len(deduped):,} (removed {len(drop_ids):,})")
    return deduped, pd.DataFrame(audit_rows)

# news_df, near_dedup_audit_df = near_dedup(news_df)
# display(near_dedup_audit_df.sort_values("jaccard", ascending=False).head(20))
```

Ghi chú thiết kế:
- `lsh.query(mh)` trước khi `lsh.insert(...)` để không tự khớp với chính nó, và để mỗi cặp chỉ được
  xét một lần theo thứ tự xử lý (giữ độ phức tạp gần O(N), tránh O(N²)).
- `jac = mh.jaccard(candidate_minhash)` tính lại Jaccard ước lượng đầy đủ từ MinHash — vì LSH banding
  là xấp xỉ xác suất, đôi khi trả về candidate có Jaccard ước lượng thấp hơn threshold (false
  positive của banding). Đây chính là dòng `REJECT_LOW_JACCARD` phục vụ yêu cầu audit false-positive
  trong báo cáo.
- Gom cụm bằng `networkx.connected_components` thay vì Union-Find viết tay — tái dùng dependency có
  sẵn, giữ cell tự chứa (không phụ thuộc `UF` class định nghĩa ở cell 2.2 phía sau).
- Đại diện giữ lại mỗi cụm: bản có `text` dài nhất (giả định bản đầy đủ nhất ít khả năng bị cắt cụt).
- Verify: import cell, gọi thử `near_dedup` trên DataFrame giả lập nhỏ (script rời, đã chạy trong venv
  cô lập, không commit vào repo) với:
  1. Hai bài giống hệt nhau → phải gộp. ✅ Đã xác nhận: jaccard=1.0, `MERGE_NEAR_DUP`.
  2. Hai bài gần giống (repost, đổi vài từ) trên văn bản **độ dài thực tế (~150 từ)** → gộp khi Jaccard
     ước lượng ≥ 0.8. ✅ Đã xác nhận: jaccard≈0.86, `MERGE_NEAR_DUP`.
  3. Hai bài không liên quan → không gộp. ✅ Đã xác nhận: bài Apple vẫn tách riêng.
  4. **Phát hiện quan trọng khi test:** với đoạn văn bản ngắn (~35 từ), chỉ 1 từ bị đổi đã kéo Jaccard
     thật xuống 0.75 (dưới threshold 0.8) — vì k=5 shingle trên văn bản ngắn khiến 1 từ đổi làm hỏng tỉ
     lệ shingle lớn hơn nhiều so với bài dài ~150-200+ từ thực tế trong dataset HackerNoon. Đây là insight
     nên đưa vào báo cáo (mục "threshold, vì sao") — threshold 0.8 phù hợp với độ dài bài báo thực tế,
     nhưng sẽ kém nhạy hơn với đoạn text rất ngắn.
  5. Cũng quan sát được: LSH banding có thể **miss** (false negative) một cặp có Jaccard ước lượng ngay
     sát ngưỡng (ví dụ 0.828 vs threshold 0.8) do bản chất xác suất của band/row config — không phải lỗi
     code. Ghi lại làm ví dụ cho phần "trade-off/hạn chế" trong báo cáo nếu cần.

## Task 3 — Wire vào pipeline
Sửa cell 1.5 (id `adb244b9`), đoạn cuối:
```python
# raw_df = load_news(DATA_PATH)
# news_df = standardize_news(raw_df)
# news_df, near_dedup_audit_df = near_dedup(news_df)
# chunks_df = build_chunks(news_df)
# display(chunks_df.head())
```
Verify: grep xác nhận thứ tự `standardize_news` → `near_dedup` → `build_chunks`.

## Task 4 — Kiểm tra tính hợp lệ của notebook
- Sau khi chỉnh JSON của .ipynb bằng tay (thêm cell mới), chạy `python -c "import json; json.load(open(path))"`
  để đảm bảo file vẫn là JSON hợp lệ (nbformat).
- Chạy `python -m py_compile` (hoặc `ast.parse`) trên source code của cell mới (trích xuất ra .py tạm) để
  đảm bảo cú pháp Python hợp lệ trước khi ghi vào notebook.

## Task 5 — Không tự động điền báo cáo
Không sửa `reports/lab_report.md` (học viên tự điền phần bonus theo yêu cầu ASSIGNMENT.md). Plan này ghi
lại sẵn 3 nội dung cần điền vào báo cáo để học viên dùng:
1. **Threshold:** 0.8 Jaccard trên 5-word shingle (chuẩn phổ biến cho phát hiện repost/near-dup báo chí;
   đủ chặt để tránh gộp nhầm 2 bài cùng chủ đề nhưng khác nội dung).
2. **False positive:** ví dụ cụ thể lấy từ `near_dedup_audit_df[decision == "REJECT_LOW_JACCARD"]` sau khi
   chạy trên dữ liệu thật — LSH banding trả về candidate nhưng Jaccard ước lượng đầy đủ < threshold.
3. **Audit:** bảng `near_dedup_audit_df` (cột `article_id_a`, `article_id_b`, `jaccard`, `decision`),
   xem tương tự cách `entity_resolution_audit_df` được audit ở Phần 5.1.

## Out of scope
- Không thay đổi cơ chế exact-hash dedup hiện có (`dedup_key` SHA-1) — near-dedup chạy **sau**, bổ sung
  chứ không thay thế.
- Không tự động chạy toàn bộ notebook trên dữ liệu thật (cần Colab/Neo4j/API keys); chỉ verify cú pháp +
  test đơn vị nhỏ cục bộ.
