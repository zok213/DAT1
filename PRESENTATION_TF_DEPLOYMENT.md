# 🎤 Kịch Bản Thuyết Trình: Triển Khai Edge AI với TensorFlow (Edge Deployment)

> **Mục đích**: Kịch bản thuyết trình được thiết kế DÀNH RIÊNG cho phần Deployment trong môn học về TensorFlow. Kịch bản này tập trung hoàn toàn vào kỹ thuật tối ưu hóa mô hình (Quantization), TensorFlow Lite (TFLite), TFLite Delegates (Hexagon DSP), và quản lý bộ nhớ Zero-Copy bằng C++.
> **Ngôn ngữ**: Tiếng Việt (Phong cách Kỹ sư AI chuyên nghiệp, tự tin, báo cáo học thuật).

---

## Slide 1: Tiêu đề & Đặt Vấn Đề (The Edge Deployment Challenge)
**Visual**: Sơ đồ từ Cloud (TensorFlow) chuyển xuống Edge (Thiết bị IoT/Solar).
**Speaker Script**:
> "Chào thầy và các bạn. Hôm nay nhóm em xin trình bày về bài toán Triển khai mô hình AI trên thiết bị biên (Edge Deployment) sử dụng hệ sinh thái TensorFlow. 
> 
> Trong môn học này, chúng ta đã quen với việc train các mô hình lớn bằng TensorFlow/Keras trên Cloud với GPU mạnh mẽ. Tuy nhiên, khi đưa mô hình (YOLOv8 và DINOv2) xuống một thiết bị chạy bằng pin mặt trời ngoài trang trại, chúng ta không thể bê nguyên code Python và file mô hình FP32 xuống được. Thiết bị sẽ quá nhiệt, nghẽn cổ chai bộ nhớ, và sập nguồn.
> 
> Phần trình bày này sẽ giải quyết câu hỏi: Làm sao để tối ưu một hệ thống thị giác máy tính nặng nề xuống chạy mượt mà ở tốc độ 30 FPS với mức tiêu thụ điện chỉ dưới 3 Watts bằng TensorFlow Lite?"

---

## Slide 2: Pipeline của hệ thống & Nút thắt cổ chai
**Visual**: Flowchart: YOLOv8 (Detection) -> Crop -> DINOv2 (BCS Head). Có một biểu tượng cảnh báo "Memory Bottleneck" ở giữa.
**Speaker Script**:
> "Hệ thống của chúng em gồm 2 mô hình liên tiếp: YOLOv8 để cắt ảnh con bò, và DINOv2 ViT để đánh giá điểm cơ thể (BCS). 
> 
> Nút thắt lớn nhất khi deploy không phải là sức mạnh tính toán, mà là băng thông bộ nhớ (Memory Bandwidth). Nếu dùng code Python thông thường, mỗi khung hình từ camera sẽ phải copy từ RAM của CPU sang GPU, rồi lại copy ngược về để cắt ảnh, sau đó lại đẩy vào NPU. Quá trình copy dữ liệu liên tục này sẽ triệt tiêu hoàn toàn hiệu năng của thiết bị, làm cho tốc độ tụt xuống chỉ còn 2 FPS.
> 
> Vì vậy, chúng em đã từ bỏ Python và chuyển hoàn toàn sang C++ Edge API."

---

## Slide 3: TensorFlow Lite & Tối ưu hóa mô hình (Quantization)
**Visual**: Code snippet của `tf.lite.TFLiteConverter` và bảng so sánh FP32 vs INT8 (Độ chính xác / Kích thước).
**Speaker Script**:
> "Bước đầu tiên trong quy trình deploy là chuyển đổi mô hình từ TensorFlow SavedModel sang định dạng TensorFlow Lite (TFLite). Định dạng này được thiết kế tối giản để chạy trên các vi xử lý nhỏ.
> 
> Đặc biệt, chúng em áp dụng kỹ thuật Post-Training Quantization (PTQ) tích hợp sẵn trong TFLite Converter. Bằng cách cung cấp một tập dữ liệu đại diện (Representative Dataset), TFLite sẽ ánh xạ các trọng số từ dấu phẩy động 32-bit (FP32) xuống số nguyên 8-bit (INT8). 
> 
> Kết quả là? Kích thước mô hình giảm đi 4 lần, băng thông bộ nhớ cần thiết giảm 4 lần, trong khi mức độ sụt giảm độ chính xác (Accuracy drop) gần như bằng 0."

---

## Slide 4: Tăng tốc phần cứng với TFLite Delegates
**Visual**: Sơ đồ TFLite Delegates: CPU (XNNPACK), GPU (OpenCL), DSP (Hexagon Delegate).
**Speaker Script**:
> "Có được file `.tflite` INT8 là chưa đủ. Nếu chỉ chạy trên CPU, hiệu năng vẫn rất tệ. Điểm sáng giá nhất của hệ sinh thái TensorFlow Lite chính là cơ chế **Delegates**.
> 
> Thay vì tự tính toán, TFLite có thể ủy quyền (delegate) các phép toán ma trận cho phần cứng chuyên dụng. Cụ thể trên bo mạch Qualcomm RB3 Gen2, chúng em sử dụng **Hexagon DSP Delegate**. Hexagon là một bộ xử lý tín hiệu số (DSP) sinh ra chỉ để nhân ma trận với mức điện năng cực thấp. 
> 
> TensorFlow Lite Delegate sẽ tự động ánh xạ các node trong đồ thị tính toán của DINOv2 vào tập lệnh của Hexagon DSP, giúp NPU chạy ở tốc độ cao nhất mà CPU vẫn hoàn toàn rảnh rỗi."

---

## Slide 5: Kiến trúc bộ nhớ Zero-Copy (C++ API)
**Visual**: Sơ đồ bộ nhớ: Camera -> Hardware Decoder -> DMA-BUF -> TFLite Tensor (Không đi qua CPU).
**Speaker Script**:
> "Để giải quyết bài toán nghẽn cổ chai bộ nhớ đã nêu ở đầu, chúng em sử dụng kỹ thuật Zero-Copy memory thông qua hệ thống `DMA-BUF` trên Linux kết hợp với TFLite C++ API.
> 
> Khi camera thu nhận hình ảnh, dữ liệu được giải mã bằng phần cứng và cấp phát một file descriptor (ION memory FD). Thay vì copy mảng byte này vào Input Tensor của TFLite, chúng em truyền thẳng con trỏ bộ nhớ (pointer) này qua TFLite Delegate.
> 
> Nghĩa là: hình ảnh chưa từng chạm vào bộ nhớ của CPU. Dữ liệu chảy thẳng từ Camera, qua bộ giải mã, vào Hexagon DSP. Nhờ kiến trúc Zero-Copy này, CPU utilization của chúng em luôn ở mức dưới 10%."

---

## Slide 6: Kết quả thực tế & Quản lý Nhiệt năng (Thermal Watchdog)
**Visual**: Biểu đồ Throughput (FPS) và Power (Watts) của Qualcomm (2.8W @ 22 FPS) so với Jetson (15W @ 31 FPS).
**Speaker Script**:
> "Với toàn bộ các kỹ thuật tối ưu từ hệ sinh thái TensorFlow: Quantization INT8, Hexagon Delegate, và Zero-Copy C++, kết quả chúng em đạt được là rất ấn tượng.
> 
> Trên bo mạch NVIDIA Jetson (giới hạn điện ở mức 15W), hệ thống chạy được 31 FPS. Nhưng đáng kinh ngạc hơn, trên bo mạch Qualcomm, TFLite Hexagon DSP xử lý mượt mà ở tốc độ **22 FPS** nhưng chỉ tiêu thụ vỏn vẹn **2.8 Watts** điện năng. Đây là thông số hoàn hảo cho một thiết bị biên chạy bằng pin năng lượng mặt trời.
> 
> Cuối cùng, chúng em tích hợp một Thermal Watchdog bằng C++. Nếu nhiệt độ của chip SoC vượt quá 75 độ C trong môi trường chuồng trại khắc nghiệt, Watchdog sẽ tự động can thiệp vào vòng lặp của TFLite, chủ động giảm FPS xuống để hạ nhiệt, ngăn chặn tình trạng tràn bộ nhớ hay sập hệ thống (kernel panic)."

---

## Slide 7: Kết Luận
**Speaker Script**:
> "Để tổng kết lại phần Deployment:
> 
> 1. Mô hình tốt trên Cloud là vô nghĩa nếu không thể chạy trên Edge. Chúng em đã dùng **TF Lite Converter** để lượng tử hóa (Quantize) mô hình xuống INT8.
> 2. Chúng em giải quyết vấn đề tính toán bằng cách tận dụng **TFLite Hexagon Delegate** để đưa toàn bộ gánh nặng ma trận sang DSP phần cứng.
> 3. Chúng em giải quyết băng thông bộ nhớ bằng C++ **Zero-Copy (DMA-BUF)**.
> 
> Qua đó, chúng em chứng minh được rằng TensorFlow không chỉ là một thư viện để train AI, mà là một giải pháp Enterprise hoàn chỉnh, đủ sức đưa các mô hình AI hạng nặng nhất xuống tận các trang trại khắc nghiệt nhất.
> 
> Cảm ơn thầy và các bạn đã lắng nghe!"
