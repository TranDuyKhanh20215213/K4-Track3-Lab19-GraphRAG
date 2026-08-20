# Số liệu trích tự động cho lab_report.md

_Sinh lúc: 2026-08-20 19:12:21_


## 1. Pipeline & phạm vi dữ liệu

- source_scope: **5,000 dòng đầu** của hackernoon_subset.csv (khớp golden dataset)
- Articles sau exact dedup: **2,118** (row_id 0..4997)
- Chunks (Flat RAG index): **2,118**
- Chunks đưa qua LLM extraction: **400** (ưu tiên evidence row của golden set)
- Near-duplicate phát hiện (MinHash/LSH @ 0.85): **18** cặp — chính sách `audit` (giữ nguyên corpus vì golden set cần cả 2 bản)
- Triples thô: **175** → sau canonicalize: **174**
- Graph: **287** nodes, **174** edges, invalid_provenance_edges = **0**

## 2. Schema quyết định

- CORE relations (theo ASSIGNMENT): **8**
- EXTENDED bật: **True** → tổng **27** relation, **5** node type
- Tỉ lệ triple thuộc CORE 8: **48.6%**

## 3. Entity Resolution

- Cosine threshold: **0.9**, lexical ratio min: **0.72**
- Audit rows: **26**

```
decision
REJECT_GUARD              13
REJECT_BELOW_THRESHOLD     5
MERGE_LEXICAL              4
MERGE_MANUAL               3
MERGE_VECTOR               1
```

**Cặp similarity cao bị Guard chặn (câu hỏi 2):**

| type         | left                                       | right                                        |   similarity | reason                      |
|:-------------|:-------------------------------------------|:---------------------------------------------|-------------:|:----------------------------|
| Organization | Middlebury Institute community             | Middlebury Institute                         |     0.925406 | REJECT_SUBBRAND_CONTAINMENT |
| Organization | Middlebury Institute community             | Middlebury Institute                         |     0.925406 | REJECT_SUBBRAND_CONTAINMENT |
| Technology   | AI software                                | AI tools                                     |     0.883637 | REJECT_LEXICAL_RATIO_0.42   |
| Company      | Fidelity National Information Services Inc | Fidelity National Information Services (FIS) |     0.86095  | REJECT_SUBBRAND_CONTAINMENT |
| Organization | Middlebury Institute community             | Middlebury College Network                   |     0.752093 | REJECT_LEXICAL_RATIO_0.50   |

## 4. Super-node

- Ngưỡng degree: **100**, edge cap: **50**, global cap: **250**, context cap: **14000** chars

**Top 3 node theo degree (câu hỏi 3):**

| name                    | type    |   degree |
|:------------------------|:--------|---------:|
| Amazon                  | Company |        6 |
| ServiceNow              | Company |        6 |
| L&T Technology Services | Company |        5 |

## 5. Độ phủ evidence của golden set

- Evidence nằm trong corpus (Flat RAG): **100%**
- Evidence đã được extract vào graph: **100%**

## 6. Benchmark

| Loại câu hỏi   | Metric              |   Flat RAG |   GraphRAG |   Delta (Graph - Flat) | Nhận xét phân tích                                       |
|:---------------|:--------------------|-----------:|-----------:|-----------------------:|:---------------------------------------------------------|
| cross-doc      | Comprehensiveness   |      3.182 |      4.545 |                  1.364 | GraphRAG cải thiện rõ; kiểm tra rationale và provenance. |
| cross-doc      | Faithfulness        |      3.182 |      4.727 |                  1.545 | GraphRAG cải thiện rõ; kiểm tra rationale và provenance. |
| cross-doc      | Multi-hop reasoning |      3.182 |      4.545 |                  1.364 | GraphRAG cải thiện rõ; kiểm tra rationale và provenance. |
| cross-doc      | Latency (s)         |      1.927 |      4.596 |                  2.669 | GraphRAG tốn thêm 139% — chi phí của subgraph context.   |
| cross-doc      | Token usage         |    717.273 |    875.909 |                158.636 | GraphRAG tốn thêm 22% — chi phí của subgraph context.    |
| factoid        | Comprehensiveness   |      5     |      5     |                  0     | Hai phương pháp gần nhau.                                |
| factoid        | Faithfulness        |      5     |      5     |                  0     | Hai phương pháp gần nhau.                                |
| factoid        | Multi-hop reasoning |      5     |      5     |                  0     | Hai phương pháp gần nhau.                                |
| factoid        | Latency (s)         |      1.227 |      3.134 |                  1.907 | GraphRAG tốn thêm 156% — chi phí của subgraph context.   |
| factoid        | Token usage         |    762.5   |    873     |                110.5   | GraphRAG tốn thêm 14% — chi phí của subgraph context.    |
| multi-hop      | Comprehensiveness   |      3.333 |      3.667 |                  0.333 | Hai phương pháp gần nhau.                                |
| multi-hop      | Faithfulness        |      3.417 |      3.667 |                  0.25  | Hai phương pháp gần nhau.                                |
| multi-hop      | Multi-hop reasoning |      3.333 |      3.583 |                  0.25  | Hai phương pháp gần nhau.                                |
| multi-hop      | Latency (s)         |      2.95  |      4.978 |                  2.028 | GraphRAG tốn thêm 69% — chi phí của subgraph context.    |
| multi-hop      | Token usage         |    751.583 |    834.083 |                 82.5   | GraphRAG tốn thêm 11% — chi phí của subgraph context.    |
| ALL            | Comprehensiveness   |      3.4   |      4.16  |                  0.76  | GraphRAG cải thiện rõ; kiểm tra rationale và provenance. |
| ALL            | Faithfulness        |      3.44  |      4.24  |                  0.8   | GraphRAG cải thiện rõ; kiểm tra rationale và provenance. |
| ALL            | Multi-hop reasoning |      3.4   |      4.12  |                  0.72  | Hai phương pháp gần nhau.                                |
| ALL            | Latency (s)         |      2.362 |      4.662 |                  2.3   | GraphRAG tốn thêm 97% — chi phí của subgraph context.    |
| ALL            | Token usage         |    737.36  |    855.6   |                118.24  | GraphRAG tốn thêm 16% — chi phí của subgraph context.    |

## 7. Ca lỗi điển hình (câu hỏi 4)

| case                  | id       | group     | question                                                                                                                                                                |   flat_score |   graph_score |     delta |
|:----------------------|:---------|:----------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------:|--------------:|----------:|
| GRAPH_WINS_FLAT_FAILS | G5000-33 | cross-doc | Which July OpenAI-related event is a content/technology collaboration, and which July event is a voluntary governance commitment?                                       |            1 |       5       |  4        |
| GRAPH_WINS_FLAT_FAILS | G5000-43 | cross-doc | Which came first in the selected HPE timeline: the Axis Security acquisition agreement or the LLM-focused cloud service announcement, and what does that ordering show? |            2 |       5       |  3        |
| GRAPH_WINS_FLAT_FAILS | G5000-45 | cross-doc | Rows 261 and 891 describe L&T Technology Services and Qualcomm being selected by Thales. How should a production graph avoid double-counting this?                      |            2 |       5       |  3        |
| GRAPH_FAILS           | G5000-42 | multi-hop | Starting from the edge-computing concept, connect one Dell product and one HPE transaction to two different edge-related outcomes.                                      |            5 |       2       | -3        |
| GRAPH_FAILS           | G5000-30 | multi-hop | Meta appears in two different AI contexts in the selected data. What are they, and what distinct relation should the graph store in each case?                          |            2 |       1       | -1        |
| GRAPH_FAILS           | G5000-27 | cross-doc | How should the graph reconcile the statement that AMD powers multiple cloud services with the later Reuters report about AWS considering AMD AI chips?                  |            4 |       3.33333 | -0.666667 |

## 8. Chi phí LLM

```
{'extract:openai/gpt-4o-mini': Counter({'total_tokens': 183152, 'prompt_tokens': 118372, 'completion_tokens': 64780, 'calls': 181}), 'answer:openai/gpt-4o-mini': Counter({'total_tokens': 45111, 'prompt_tokens': 38006, 'completion_tokens': 7105, 'calls': 82}), 'judge:openai/gpt-4o-mini': Counter({'total_tokens': 53536, 'prompt_tokens': 47812, 'completion_tokens': 5724, 'calls': 52})}
```