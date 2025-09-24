# Phat-hien-gian-lan-thi-cu-truc-tuyen-bang-thi-giac-may-tinh
Phát hiện gian lận thi cử trực tuyến bằng thị giác máy tính

<img width="1356" height="1080" alt="image" src="https://github.com/user-attachments/assets/23394fb2-1c94-462c-8b8b-8cecdaf4e4e4" />

894: Mô hình dự đoán đúng “cheating” (True Positive).
1051: Mô hình dự đoán đúng “non-cheating”.
495: Mô hình dự đoán đúng “background”.

482: trường hợp thật ra là cheating, nhưng mô hình lại dự đoán thành background.
193: trường hợp thật ra là cheating, nhưng bị dự đoán nhầm thành non-cheating.
495: trường hợp thật ra là background, nhưng bị gán nhãn non-cheating.


<img width="2132" height="1224" alt="image" src="https://github.com/user-attachments/assets/0634d162-c6c0-453d-951b-9e9338765243" />

Loss giảm ổn định → không có dấu hiệu overfitting.
Precision & Recall ~65% → mô hình ở mức khá.mAP50 = 0.65 → chấp nhận được.
mAP50-95 = 0.35 → mô hình chưa thật sự mạnh ở ngưỡng khắt khe (IoU cao).


Cách chạy code
Mở PowerShell tại thư mục chứa app.py, chạy:
uvicorn app:app --reload
