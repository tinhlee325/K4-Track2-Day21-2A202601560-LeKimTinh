# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

|              |                                                                     |
| ------------ | ------------------------------------------------------------------- |
| Họ và tên | Lê Kim Tính                                                       |
| MSSV         | 2A202601560                                                         |
| Lớp / Khóa | K4                                                                  |
| Repo GitHub  | https://github.com/tinhlee325/K4-Track2-Day21-2A202601560-LeKimTinh |
| Ngày nộp   | 21/08/2026                                                          |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
| ---------- | ------------ | ------------- | --------- | -------- | -------- |
| 1          | 200          | 0.1           | 5         | 0.7149   | 0.8740   |
| 2          | 100          | 0.1           | 3         | 0.7109   | 0.8780   |
| 3          | 50           | 0.05          | 2         | 0.6051   | 0.8460   |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Bộ tham số này đem lại giá trị `f1_score` cao nhất (0.7149) và vượt qua ngưỡng kiểm định chất lượng 0.65. Đáng chú ý, lần chạy 2 đạt `accuracy` cao nhất (0.8780) nhưng `f1_score` lại thấp hơn lần 1 (0.7109), cho thấy độ chính xác tổng thể có thể bị chi phối bởi lớp đa số. Khi tăng số lượng cây (`n_estimators=200`) kết hợp độ sâu cây hợp lý (`max_depth=5`), mô hình Gradient Boosting học được các mối quan hệ phi tuyến phức tạp tốt hơn, cải thiện khả năng phân loại chính xác các mẫu thu nhập cao. Giữa `n_estimators` và `learning_rate` có sự đánh đổi rõ rệt: giảm tốc độ học đòi hỏi phải tăng số lượng cây để bù đắp dung lượng học của mô hình.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Adult có phân bố lớp mất cân bằng nghiêm trọng khi chỉ khoảng 24.8% số mẫu thuộc lớp thu nhập cao (>50K USD/năm). Nếu xây dựng một mô hình ngây thơ luôn dự đoán nhãn "thu nhập thấp" cho tất cả mọi người, mô hình đó vẫn đạt Accuracy lên tới 75.2%, tạo ra ảo giác về một mô hình tốt dù nó hoàn toàn vô dụng và không bắt được bất kỳ trường hợp thu nhập cao nào. Ngược lại, chỉ số `f1_score` tính trung bình điều hòa giữa Precision và Recall của riêng lớp dương (target = 1), phản ánh trung thực khả năng nhận diện lớp thiểu số quan trọng này. Khi gọi hàm đánh giá, ta không sử dụng `average="weighted"` hay `average="macro"` vì các trọng số này sẽ bị lớp đa số kéo lên cao, làm mất đi ý nghĩa cảnh báo của ngưỡng chất lượng (Quality Gate).

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn                                                             | Nguyên nhân                                                                                       | Cách giải quyết                                                                      |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Lỗi kết nối SSH và định dạng private key trên Windows          | PowerShell lưu file key theo mã hóa UTF-16LE làm OpenSSH báo lỗi invalid format               | Chuyển đổi định dạng file key sang chuẩn mã hóa ASCII/UTF-8 không BOM         |
| Lỗi unpickle mô hình khi khởi động service trên EC2             | Phiên bản`scikit-learn` trên máy ảo EC2 mới hơn phiên bản 1.4.2 của môi trường CI/CD | Cài đặt cố định chính xác phiên bản`scikit-learn==1.4.2` trên máy ảo EC2 |
| Lỗi thiếu cấu hình MLflow tracking URI trên runner GitHub Actions | Runner GitHub Actions là môi trường sạch, chưa có sẵn thư mục`mlruns` mặc định       | Cấu hình mặc định tracking URI sang SQLite cục bộ trong mã nguồn và workflow  |

---

## 4. So Sánh Bước 2 và Bước 3

|                                  | f1_score | accuracy |
| -------------------------------- | -------- | -------- |
| Bước 2 (chỉ`train_batch1`)  | 0.7149   | 0.8740   |
| Bước 3 (thêm`train_batch2`) | 0.7354   | 0.8820   |

**Nhận xét:** Khi bổ sung thêm 22.361 mẫu dữ liệu mới ở Bước 3 (tổng cộng 44.722 mẫu), điểm F1 tăng nhẹ từ 0.7149 lên 0.7354 và Accuracy tăng từ 0.8740 lên 0.8820. Do dữ liệu mới được phân chia từ cùng nguồn và cùng phân phối, sự cải thiện không quá đột biến nhưng việc bổ sung thêm mẫu giúp mô hình tổng quát hóa tốt hơn. Điều cốt lõi được chứng minh là toàn bộ quy trình MLOps đã vận hành hoàn toàn tự động: chỉ từ một commit cập nhật dữ liệu, pipeline CI/CD đã tự động kéo dữ liệu, huấn luyện lại, vượt qua Quality Gate và tái triển khai thành công lên máy chủ sản xuất.
