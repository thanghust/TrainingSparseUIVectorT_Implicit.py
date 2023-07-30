#Giới thiệu

Bài toán đề cập trong bài code này là hệ thống gợi ý sản phẩm sử dụng phương pháp TIME-SVD++. Đây là một trong những phương pháp phổ biến được sử dụng để giải quyết bài toán hệ thống gợi ý. Phương pháp này mở rộng phương pháp SVD++ theo thời gian, nghĩa là sử dụng thông tin thời gian khi tính toán.

Bài toán giới thiệu gồm 3 thành phần chính: người dùng, sản phẩm và thời gian. Mục tiêu của hệ thống là dự đoán xem người dùng sẽ đánh giá sản phẩm nào vào thời điểm nào.

Bài code này sử dụng ngôn ngữ Python và thư viện TensorFlow để triển khai phương pháp TIME-SVD++. Dữ liệu được sử dụng để huấn luyện là tập dữ liệu sản phẩm trên Amazon.

#Hướng dẫn sử dụng

* Để sử dụng code,  cần cài đặt môi trường TensorFlow, cài các thư viện như matplotlib, pandas, numpy
* Chạy chương trình chính trên file **AmazonRecommender.ipynb**, chạy lần lượt các cell
* Để huấn luyện mô hình TIMESVD++, cần gọi hàm 'train_by_df_timeVCDplus' trong file **AmazonRecommender.ipynb** và truyền các config đầu vào. Trong quá trình huấn luyện , mô hình sẽ lưu trọng số tại các epoch tốt nhất của tập train và đánh giá trên tập dev để  có thể sử dụng lại đánh gia trên tập test. Kết quả đánh giá được in ra sau mỗi epoch.
**Lưu ý**: 'train_by_df()' cũng sẽ có những config giống như hàm 'train_by_df_timeVCDplus()' nhưng sẽ chạy với những phương pháp không có implicit feedback ( chỉ là sự thay đổi của bias user hoặc item, hoặc sự thay đổi của nhân tố ẩn user). Kết quả của phương pháp khác sẽ dùng để so sánh hiệu quả với phương pháp TIMESVD++
