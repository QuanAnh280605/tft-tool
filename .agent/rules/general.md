---
trigger: always_on
description: General rules for TFT Tool project development
globs: "**/*"
---

## Dependency Management

- **BẮT BUỘC**: Mỗi khi cài thêm thư viện Python mới vào dự án, **PHẢI** thêm tên thư viện đó vào file `requirements.txt` ngay lập tức.
- Không được sử dụng thư viện mà chưa khai báo trong `requirements.txt`.
- Khi xóa thư viện không dùng nữa, cũng phải xóa khỏi `requirements.txt`.

## Project Structure

- Tách logic nghiệp vụ thành các module riêng biệt (ví dụ: `ocr_service.py`, `detect_shop.py`).
- File `main.py` chỉ chứa route definitions và orchestration logic.
- Sử dụng singleton pattern cho các heavy resources (ML models, database connections).
