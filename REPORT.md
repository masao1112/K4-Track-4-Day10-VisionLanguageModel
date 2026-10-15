Họ tên: Thai Anh
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | - | - | 0.084 | [0.060, 0.111] | 0.98 | 0.152 | 0.027 | 0.161 |
| E1 | LoRA LLM | 33,030,144 | 4.32 | 5.63 | 0.234 | [0.200, 0.274] | 1.32 | 0.127 | 0.170 | 0.486 |
| E2 | + projector | 142,135,296 | 11.04 | 5.68 | 0.233 | [0.200, 0.273] | 1.09 | 0.142 | 0.180 | 0.417 |
| E3 | + LoRA ViT | 148,426,752 | 11.90 | 7.03 | 0.260 | [0.221, 0.298] | 1.04 | 0.137 | 0.194 | 0.520 |
| E4 | Ablation rank (r=4 / r=64) | 8,257,536 / 132,120,576 | 8.99 / 11.07 | 6.30 / 6.59 | 0.237 / 0.248 | [0.199, 0.273] / [0.211, 0.286] | 1.19 / 1.42 | 0.151 / 0.184 | 0.158 / 0.164 | 0.506 / 0.529 |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
- **Quy mô tham số từng khối**:
  - **LLM + text embedding**: Khoảng 4,02B tham số (36 transformer layers, mỗi layer gồm self-attention `{q, k, v, o}_proj` và MLP `{gate, up, down}_proj`).
  - **ViT (Vision Transformer)**: Khoảng 415M tham số (24 visual blocks với `{qkv, proj}` và MLP `{linear_fc1, linear_fc2}`).
  - **Projector**: Nằm trong module vision, gồm `visual.merger` (khoảng 27M tham số) và `visual.deepstack_merger_list` (khoảng 82M tham số, kết nối đặc trưng đa tầng ở các block 5, 11, 17 của ViT). Tổng tham số của projector là 109,105,152 tham số (~109M).
  - Tổng số tham số toàn mô hình Qwen3-VL-4B là ~4,44B.
- **Khối được cập nhật trong từng setup**:
  - **E1**: Chỉ gắn LoRA ($r=16, \alpha=32$) vào LLM (252 modules: 36 layers × 7 modules). Toàn bộ ViT và projector bị đóng băng hoàn toàn. Tổng tham số huấn luyện: **33,030,144** tham số.
  - **E2**: Giữ nguyên LoRA LLM như E1, đồng thời mở đầy đủ (full fine-tune) toàn bộ projector (`visual.merger` và `visual.deepstack_merger_list`) ở kiểu dữ liệu `float32`. Tổng tham số huấn luyện: **142,135,296** tham số (tăng thêm đúng 109,105,152 tham số projector).
  - **E3**: Giữ nguyên toàn bộ cấu hình của E2 (LoRA LLM + full projector), đồng thời bổ sung thêm LoRA ($r=16, \alpha=32$) vào ViT trên 96 modules (24 blocks × 4 modules attention và MLP). Tổng tham số huấn luyện: **148,426,752** tham số (tăng thêm 6,291,456 tham số LoRA vision).

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
- **Tỷ lệ parse hợp lệ**: `parse_valid_rate = 0.9035` (90.35%, có 281/311 mẫu trả lời đúng cú pháp JSON list mong muốn).
- **Hai lỗi phổ biến nhất của mô hình gốc zero-shot**:
  1. *Ảo giác bounding box khổng lồ / sai lệch vùng không gian*: Mô hình dự đoán các box toạ độ bao trùm gần như toàn bộ nửa dưới mặt đường thay vì khoanh vùng vết nứt cục bộ (ví dụ: `[0, 700, 998, 998]`).
  2. *False Positives nghiêm trọng và nhận diện sai lớp*: Số lượng box bịa đặt rất lớn (FP/ảnh = 0.98, tổng cộng 306 FP trên 311 ảnh). Đặc biệt lớp `crack` có Precision chỉ đạt 0.031 (3.1%) và Recall chỉ 0.023 do mô hình nhầm lẫn hầu hết vạch kẻ đường, rãnh phân làn, bóng râm thành hư hại đường.
- **Ví dụ output thô thực tế** (trích từ ảnh `vlcsnap-00201`):
  ```json
  [{"label": "crack", "bbox_2d": [0, 700, 998, 998]}]
  ```
  Box này bao trùm toàn bộ chiều rộng từ x=0 đến x=998 và chiều cao từ y=700 đến y=998, hoàn toàn không khớp với vết nứt nhỏ thực tế khiến IoU < 0.1 và bị tính là False Positive.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
- **Kết quả F1 khi lệch thang toạ độ**: F1 sẽ lập tức rơi về **0.000** (Precision = 0, Recall = 0, TP = 0).
- **Nguyên nhân**:
  - Mô hình Qwen3-VL được thiết kế để xuất toạ độ chuẩn hoá theo thang `[0, 1000]` tương đối so với kích thước khung hình (grounding convention).
  - Với ảnh dashcam có kích thước thực tế là 640×360:
    - Nếu nhãn hoặc công cụ chấm sử dụng toạ độ pixel tuyệt đối ($x \in [0, 640], y \in [0, 360]$) trong khi mô hình xuất toạ độ thang 1000 (ví dụ toạ độ tâm x=500, y=500), các toạ độ dự đoán sẽ vượt hoàn toàn ra ngoài hoặc lệch pha hoàn toàn so với box thực tế.
    - Ngược lại, nếu mô hình xuất theo pixel ($x \in [0, 640]$) mà công cụ chấm dùng thang 1000, box dự đoán sẽ bị co cụm về góc trên bên trái khung hình.
  - Khi diện tích giao nhau giữa box dự đoán và ground truth không đạt ngưỡng $IoU \ge 0.5$, không có bất kỳ cặp box nào được ghép đôi thành công $\Rightarrow$ True Positives = 0, dẫn tới F1 = 0.
- **Cách kiểm chứng**:
  - Trong quá trình triển khai Bước 2 của lab, hàm chuyển đổi `yolo_to_box` được kiểm thử đảo chiều: chuyển nhãn YOLO sang toạ độ thang 1000, sau đó nhân ngược lại tỷ lệ ảnh pixel ($w=640, h=360$) để đối chiếu với bounding box gốc; hai box hoàn toàn trùng khớp từng pixel.
  - Khi chạy thử nghiệm đối chiếu `score.py`, việc thay đổi tuỳ chọn `--coord pixel` trên file dự đoán thang `norm1000` khiến F1 lập tức giảm từ 0.234 về 0.000 do toàn bộ IoU rơi xuống dưới 0.1.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
- **Vì sao chỉ tính loss trên phần trả lời**:
  - Bản chất của bài toán là sinh chuỗi có điều kiện (conditional generation) $P(Y|X)$, trong đó $X$ là ảnh đầu vào kết hợp câu lệnh hướng dẫn (prompt), còn $Y$ là chuỗi token JSON chứa nhãn và toạ độ box.
  - Mục tiêu tối ưu là rèn luyện mô hình đưa ra câu trả lời chính xác $Y$. Bằng cách gán `labels = -100` cho toàn bộ các token thuộc về prompt và 220 visual tokens của ảnh, hàm mất mát `CrossEntropyLoss` của PyTorch sẽ bỏ qua hoàn toàn các vị trí này (`ignore_index = -100`), chỉ tập trung tính gradient trên các token của câu trả lời JSON (và token kết thúc `<|im_end|>`).
- **Điều gì xảy ra nếu tính loss trên cả prompt và token ảnh**:
  1. *Lãng phí dung lượng học*: Prompt chiếm hàng chục token cố định và ảnh chiếm tới 220 token. Nếu tính loss trên toàn bộ, phần lớn gradient bị chi phối bởi việc học tái tạo lại các token prompt lặp đi lặp lại và token hình ảnh thay vì học bài toán phát hiện vị trí hư hại.
  2. *Làm loãng tín hiệu định vị*: Gradient truyền về các trọng số LoRA bị nhiễu bởi các token ngữ cảnh, khiến mô hình khó hội tụ, khả năng grounding suy giảm nghiêm trọng.
  3. *Mất ổn định khi huấn luyện*: Các token ảnh được đưa qua mô hình dưới dạng embedding đặc trưng, việc ép hàm causal language modeling dự đoán autoregressive trên chính các token này dễ dẫn đến tràn số (loss nan/inf trong fp16) hoặc làm suy giảm năng lực biểu diễn đa phương thức sẵn có của pretrained model.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
- **Vì sao không thể huấn luyện nếu ViT bị nén 4-bit**:
  - Projector trong Qwen3-VL gồm `visual.merger` và `visual.deepstack_merger_list`, là các module tuyến tính nằm bên trong phân cấp của khối `visual`.
  - Nếu lượng tử hoá toàn bộ mạng theo 4-bit (NF4), các module này sẽ được nạp dưới dạng `Linear4bit` đóng băng với trọng số đã bị lượng tử hoá và lưu trữ dạng nén. Ta không thể duy trì gradient liên tục hay cập nhật trực tiếp trọng số đầy đủ của các tham số này mà không giải nén.
- **Cách xử lý**:
  - Cấu hình `BitsAndBytesConfig` với tham số `llm_int8_skip_modules=["visual", "lm_head"]`. Tùy chọn này giữ toàn bộ khối ViT và projector ở độ chính xác dấu phẩy động ban đầu (fp16), chỉ lượng tử hoá 4-bit trên các tầng transformer của LLM. Nhờ đó, các module của projector sẵn sàng cho việc mở gradient (`requires_grad = True`).
- **Vì sao projector phải ở float32**:
  - Projector đóng vai trò nén và chuyển đổi chiều không gian vector từ ViT sang LLM, nằm ngay tại giao điểm nơi nhận luồng gradient tích lũy từ 36 tầng LLM truyền ngược về ViT.
  - Trong quá trình cập nhật đầy đủ (full fine-tuning) với gradient accumulation và AdamW optimizer, dải giá trị của gradient tại đây dao động mạnh. Nếu để ở `float16` (chỉ có 5 bit exponent, giá trị cực đại ~65504), phép tính rất dễ bị underflow hoặc overflow dẫn đến loss bị `nan`/`inf`.
  - Việc ép kiểu tường minh toàn bộ tham số projector sang `torch.float32` đảm bảo độ chính xác số học cao, độ ổn định tuyệt đối trong suốt quá trình backpropagation.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
- **Vì sao chia ngẫu nhiên cho F1 cao giả**:
  - Tập dữ liệu đường sá được trích xuất từ các frame video liên tiếp của dashcam hành trình (quay cách nhau 1-2 giây). Các frame liền kề trong cùng một phiên quay (`session`) có bối cảnh mặt đường, ánh sáng, góc camera và các vết nứt/ổ gà gần như giống hệt nhau.
  - Nếu chia ngẫu nhiên từng ảnh (random split), các frame của cùng một vị trí sẽ xuất hiện ở cả tập train và tập test/val $\Rightarrow$ gây ra hiện tượng rò rỉ dữ liệu (data leakage) nghiêm trọng.
  - Mô hình sẽ "học vẹt" (memorize) đặc trưng nền đường cụ thể thay vì học đặc trưng tổng quát của hư hại. Khi đánh giá trên test ngẫu nhiên, F1 sẽ cao giả tạo nhưng sẽ giảm mạnh khi triển khai trên cung đường mới ngoài thực tế. Gom toàn bộ một phiên quay vào duy nhất một split loại bỏ hoàn toàn nguy cơ này.
- **Số ảnh và số box từng lớp ở mỗi split** (chuẩn hoá theo `meta.json`):
  - **Train**: 1,422 ảnh · pothole: 1,074 · crack: 1,750 · manhole: 691 (Tổng cộng: 3,515 box).
  - **Val**: 276 ảnh · pothole: 66 · crack: 470 · manhole: 103 (Tổng cộng: 639 box).
  - **Test**: 311 ảnh · pothole: 121 · crack: 299 · manhole: 163 (Tổng cộng: 583 box).
  *Nhận xét*: Do phân chia theo phiên quay tự nhiên, tỷ lệ phân bố giữa các split không hoàn toàn đồng đều (tập val có rất ít pothole: chỉ 66 box so với 470 crack), phản ánh đúng thực tế thu thập dữ liệu ngoài hiện trường.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
- **Số liệu đối chiếu thực tế**:
  - **E1 (LoRA LLM)**: F1 = **0.234**, CI95 = `[0.200, 0.274]`, VRAM đỉnh = **4.32 GB**, Thời gian huấn luyện = **5.63 phút**.
  - **E2 (+ Projector)**: F1 = **0.233**, CI95 = `[0.200, 0.273]`, VRAM đỉnh = **11.04 GB**, Thời gian huấn luyện = **5.68 phút**.
  - **E3 (+ LoRA ViT)**: F1 = **0.260**, CI95 = `[0.221, 0.298]`, VRAM đỉnh = **11.90 GB**, Thời gian huấn luyện = **7.03 phút**.
- **Phân tích chi tiết**:
  - *Mở Projector (E2 so với E1)*: **Hoàn toàn không đáng**. F1 gần như giữ nguyên (0.234 $\rightarrow$ 0.233), CI95 chồng lấn tuyệt đối (`[0.200, 0.274]` vs `[0.200, 0.273]`), trong khi VRAM đỉnh tăng vọt gấp 2.55 lần (từ 4.32 GB lên 11.04 GB) do phải lưu trữ trạng thái bộ tối ưu AdamW (FP32) cho 109M tham số projector.
  - *Bổ sung LoRA ViT (E3 so với E2 và E1)*: **Có cải thiện tích cực về hiệu năng**. F1 tăng từ 0.234 lên 0.260 (+0.026 điểm F1), tỷ lệ parse valid đạt tuyệt đối 100%, F1 của lớp nắp cống (`manhole`) tăng lên 0.520 và `crack` đạt 0.194 (cao nhất trong các setup). VRAM chỉ tăng thêm 0.86 GB so với E2 (11.04 $\rightarrow$ 11.90 GB) và thời gian tăng thêm ~1.35 phút.
  - *Kết luận về tính hiệu quả*: Xét trên khoảng tin cậy CI95, khoảng giá trị của E1 `[0.200, 0.274]` và E3 `[0.221, 0.298]` vẫn có phần chồng lấn (overlap từ 0.221 đến 0.274). Vì vậy, **E1 là cấu hình tối ưu chi phí/hiệu năng nhất** (chạy nhẹ nhàng trên các GPU phổ thông chỉ cần >4.5 GB VRAM), trong khi **E3 là cấu hình đạt độ chính xác cao nhất** nếu người dùng có sẵn GPU 16 GB VRAM.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
- **Độ rộng của khoảng tin cậy CI95**:
  - Khoảng tin cậy 95% (tính bằng phương pháp bootstrap 1,000 lần lấy mẫu lại) có biên độ rộng trung bình khoảng **0.074 đến 0.077** (tức $\pm 0.037$ quanh giá trị trung bình).
  - Cụ thể: E1 là `[0.200, 0.274]`, E2 là `[0.200, 0.273]`, E3 là `[0.221, 0.298]`, E4 (r=4) là `[0.199, 0.273]`, E4 (r=64) là `[0.211, 0.286]`.
- **Hạn chế đối với kết luận so sánh**:
  - Tập test gồm 311 ảnh chỉ chứa tổng cộng 583 box, chia cho 3 lớp hư hại. Do cỡ mẫu tương đối nhỏ, phương sai lấy mẫu (sampling variance) còn rất lớn.
  - Khoảng tin cậy của hầu hết các cấu hình đều đè lên nhau đáng kể (ví dụ vùng giao thoa giữa E1 và E3 trải dài từ 0.221 đến 0.274).
  - Về mặt thống kê suy luận (statistical inference), ta **chưa thể bác bỏ giả thuyết không** (null hypothesis) với mức ý nghĩa $\alpha=0.05$ để khẳng định chắc chắn rằng E3 luôn vượt trội hơn E1 trên mọi phân phối thực tế, hay rank 64 chắc chắn tốt hơn rank 4. Sự cải thiện về điểm số trung bình là tín hiệu tích cực, nhưng để đưa ra kết luận có ý nghĩa thống kê phân định tuyệt đối, cần kiểm tra trên tập test quy mô hàng nghìn ảnh hoặc sử dụng kiểm định bắt cặp (paired permutation test).

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
- **Lớp có kết quả kém nhất**: `pothole` (ổ gà) có F1 thấp nhất trong các mô hình đã huấn luyện (E1=0.127, E2=0.142, E3=0.137), với Recall rất thấp (~0.099 - 0.116, tức bỏ sót gần 90% ổ gà trên tập test). Theo sau là `crack` với Precision cực thấp (0.151 - 0.196) do sinh ra quá nhiều False Positives (hơn 230 - 326 FP mỗi setup).
- **Lớp có kết quả tốt nhất**: `manhole` (nắp cống) luôn đạt điểm rất cao (F1 từ 0.417 đến 0.529, Precision đạt 0.674 - 0.771).
- **Nguyên nhân cốt lõi**:
  1. *Kích thước bounding box và độ phân giải hình học*: Ổ gà và vết nứt có kích thước vật lý rất nhỏ (diện tích trung vị chỉ chiếm ≈ 0.7% diện tích ảnh). Khi toàn bộ ảnh 640×360 bị nén thành 220 visual tokens, các đặc trưng vi mô của vết nứt và ổ gà bị mờ nhạt. Ngược lại, nắp cống có diện tích lớn hơn, cấu trúc hình học tròn/vuông rõ ràng và viền kim loại/bê tông tương phản mạnh với mặt đường.
  2. *Độ phức tạp ngữ nghĩa và sự nhầm lẫn bối cảnh*: Vết nứt (`crack`) có ranh giới phân bố hình học phức tạp (dài, mảnh, ngoằn ngoèo). Mô hình liên tục nhầm các vệt dầu loang, mối nối nhựa đường, bóng râm của cây cối hoặc dây điện thành vết nứt (dẫn tới FP/ảnh cao). Ổ gà (`pothole`) thường chìm sâu, bị đọng nước hoặc bóng tối che khuất nên mô hình dễ bỏ sót.
  3. *Mất cân bằng số mẫu*: Dù train có 1,074 pothole nhưng val chỉ có 66 và test có 121 box, trong khi các dạng hình thái của ổ gà thực tế vô cùng phong phú.
- **Ảnh minh hoạ minh chứng**:
  - Tập hợp ảnh mẫu đối chiếu các lớp nhãn và bounding box được lưu tại thư mục [`report/images/`](file:///home/masao-1112/Documents/AI20K/Track4/K4-Track-4-Day10-VisionLanguageModel/report/images):
    - [Minh hoạ tổng hợp 6 mẫu đa lớp](file:///home/masao-1112/Documents/AI20K/Track4/K4-Track-4-Day10-VisionLanguageModel/report/images/visualization_1.png)
    - [Ảnh 1 - Pothole nhỏ vùng trũng](file:///home/masao-1112/Documents/AI20K/Track4/K4-Track-4-Day10-VisionLanguageModel/report/images/img1_pothole_ground_truth.png)
    - [Ảnh 2 - Vết nứt kéo dài dễ lẫn bối cảnh](file:///home/masao-1112/Documents/AI20K/Track4/K4-Track-4-Day10-VisionLanguageModel/report/images/img2_crack_ground_truth.png)
    - [Ảnh 3 - Nắp cống viền tương phản sắc nét](file:///home/masao-1112/Documents/AI20K/Track4/K4-Track-4-Day10-VisionLanguageModel/report/images/img3_manhole_ground_truth.png)
    - [Ảnh 4 - Hư hại tổ hợp nhiều box](file:///home/masao-1112/Documents/AI20K/Track4/K4-Track-4-Day10-VisionLanguageModel/report/images/img4_complex_damage.png)
    - [Ảnh 5 - Vết rạn nứt mỏng dễ bị bỏ sót](file:///home/masao-1112/Documents/AI20K/Track4/K4-Track-4-Day10-VisionLanguageModel/report/images/img5_subtle_crack.png)
    - [Ảnh 6 - Nắp cống và ổ gà cùng xuất hiện](file:///home/masao-1112/Documents/AI20K/Track4/K4-Track-4-Day10-VisionLanguageModel/report/images/img6_pothole_manhole.png)

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
- **Biến được chọn**: Khảo sát sự thay đổi của siêu tham số LoRA `rank` ($r$) trên kiến trúc E1 với hai giá trị đối chiếu:
  - $r = 4$ ($\alpha = 8$)
  - $r = 64$ ($\alpha = 128$)
- **Kết quả đo đạc từ bằng chứng**:
  - **E4 ($r=4$)**: Tham số huấn luyện = **8,257,536** (8.26M) | VRAM đỉnh = **8.99 GB** | Thời gian = **6.30 phút** | F1 = **0.237** | CI95 = `[0.199, 0.273]` | FP/ảnh = **1.19** | Pothole F1 = **0.151**, Crack F1 = **0.158**, Manhole F1 = **0.506**.
  - **E4 ($r=64$)**: Tham số huấn luyện = **132,120,576** (132.12M) | VRAM đỉnh = **11.07 GB** | Thời gian = **6.59 phút** | F1 = **0.248** | CI95 = `[0.211, 0.286]` | FP/ảnh = **1.42** | Pothole F1 = **0.184**, Crack F1 = **0.164**, Manhole F1 = **0.529**.
- **Đánh giá và độ tin cậy của kết luận**:
  - Về mặt giá trị trung bình: Tăng rank từ 4 lên 64 giúp F1 tăng thêm **+0.011** (từ 0.237 lên 0.248), đặc biệt lớp `pothole` tăng từ 0.151 lên 0.184 và `manhole` tăng từ 0.506 lên 0.529. Tuy nhiên, cái giá phải trả là số lượng tham số huấn luyện tăng vọt **gấp 16 lần** (từ 8.26M lên 132.12M) và VRAM đỉnh tăng thêm hơn 2 GB.
  - Độ tin cậy thống kê: **Chưa đủ tin cậy để khẳng định $r=64$ luôn vượt trội hơn $r=4$**. Khoảng tin cậy CI95 của hai thiết lập (`[0.199, 0.273]` và `[0.211, 0.286]`) có vùng trùng lặp rất lớn (từ 0.211 đến 0.273). Mức chênh lệch 0.011 hoàn toàn có thể bắt nguồn từ biến thiên ngẫu nhiên trên 311 ảnh test.
  - Kết luận thực nghiệm: Với bài toán phát hiện hư hại đường sá ở tập dữ liệu này, rank nhỏ $r=4$ hoặc $r=16$ là lựa chọn hợp lý nhất để tiết kiệm bộ nhớ mà không làm suy giảm đáng kể năng lực của mô hình.

## Nhìn lại

- **Hạn chế của mô hình và hàm mất mát**:
  - Việc tối ưu VLM qua hàm mất mát sinh chuỗi Cross-Entropy thông thường đối xử với các chữ số toạ độ (ví dụ token `5` và `6` trong `500` và `600`) như các phân loại độc lập, không có cơ chế phạt khoảng cách toạ độ mượt mà như hàm Smooth L1 hay GIoU trong các mô hình thuần thị giác (YOLO, DETR).
  - 220 visual tokens là kích thước nén tương đối lớn đối với các chi tiết vi mô, khiến mô hình gặp khó khăn tự nhiên khi định vị vết nứt mảnh.
- **Kinh nghiệm về phân chia dữ liệu và đánh giá thực nghiệm**:
  - Việc phân chia theo phiên quay (`session-based split`) là bài học quý giá nhất của bài lab: nó phơi bày chính xác năng lực thực tế của mô hình khi đối mặt với cung đường mới, giúp tránh xa bẫy F1 cao ảo do rò rỉ dữ liệu frame liền kề.
  - Khoảng tin cậy CI95 là công cụ bắt buộc khi làm việc với tập kiểm thử quy mô nhỏ, giúp người làm kỹ thuật không bị đánh lừa bởi những chênh lệch số thập phân không mang ý nghĩa thống kê.
