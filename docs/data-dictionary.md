# Credit Card Customer Dataset - Data Dictionary

> Dataset gốc: 8950 khách hàng, 18 cột, phản ánh hành vi sử dụng thẻ tín dụng trong 6 tháng gần nhất.

---

## Thông tin khách hàng

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| **CUST_ID** | string | Mã định danh duy nhất (vd: C10001). **Không dùng trong phân tích**, chỉ dùng để nhận diện khách hàng. |

---

## Nhóm 1 — Số dư & Tần suất cập nhật

| Cột | Kiểu | Mô tả | Ví dụ thực tế |
|-----|------|-------|---------------|
| **BALANCE** | float | Số dư còn lại trong tài khoản để mua hàng (số tiền chưa trả). Càng cao = nợ càng nhiều. | Số dư: $1,500 → khách đang nợ ngân hàng $1,500 chưa trả |
| **BALANCE_FREQUENCY** | float (0–1) | Tần suất cập nhật số dư. 1 = cập nhật mỗi kỳ, 0 = không cập nhật. | 0.9 → khách thanh toán gần như mỗi tháng; 0.2 → hiếm khi thanh toán |

---

## Nhóm 2 — Hành vi mua hàng

| Cột | Kiểu | Mô tả | Ví dụ thực tế |
|-----|------|-------|---------------|
| **PURCHASES** | float | Tổng **số tiền** đã mua (bao gồm cả trả góp và đứt). | Tổng mua: $5,000/tháng → Purchase activity cao |
| **ONEOFF_PURCHASES** | float | **Số tiền** mua lớn nhất trong **một lần** mua đứt (không trả góp). | Mua TV $2,000 đứt một lần → ONEOFF = $2,000 |
| **INSTALLMENTS_PURCHASES** | float | **Số tiền** mua theo hình thức **trả góp**. | Mua máy $1,200 chia 6 tháng → INSTALLMENTS = $1,200 |
| **PURCHASES_FREQUENCY** | float (0–1) | Tần suất mua hàng **tổng thể** (mọi hình thức). 1 = luôn luôn có giao dịch. | 0.9 → mua gần như mỗi tháng |
| **ONEOFF_PURCHASES_FREQUENCY** | float (0–1) | Tần suất mua **đứt** (không trả góp). 1 = chỉ mua đứt. | 1.0 → luôn mua đứt; 0.2 → hiếm khi mua đứt (thường trả góp) |
| **PURCHASES_INSTALLMENTS_FREQUENCY** | float (0–1) | Tần suất mua **trả góp**. 1 = luôn trả góp. | 0.8 → thường xuyên mua trả góp |
| **PURCHASES_TRX** | int | **Số lượng giao dịch mua** (đếm giao dịch, không phải số tiền). | 45 giao dịch → khách rất active |

> **Quan hệ quan trọng:**
> - `PURCHASES` = `ONEOFF_PURCHASES` + `INSTALLMENTS_PURCHASES`
> - `PURCHASES_FREQUENCY` ≥ `ONEOFF_PURCHASES_FREQUENCY` + `PURCHASES_INSTALLMENTS_FREQUENCY`

---

## Nhóm 3 — Rút tiền mặt (Cash Advance)

> Cash Advance = rút tiền mặt từ thẻ tín dụng. Phí cao, lãi suất cao hơn mua hàng. Thường là dấu hiệu cần tiền gấp.

| Cột | Kiểu | Mô tả | Ví dụ thực tế |
|-----|------|-------|---------------|
| **CASH_ADVANCE** | float | Tổng **số tiền** đã rút mặt từ thẻ. | Rút $3,000 → khách có thể đang gặp khó khăn tài chính |
| **CASH_ADVANCE_FREQUENCY** | float (0–1) | Tần suất rút tiền mặt. 1 = rút mỗi tháng. | 0.5 → rút tiền cách tháng |
| **CASH_ADVANCE_TRX** | int | **Số lượng giao dịch** rút tiền mặt. | 8 lần rút → thường xuyên rút tiền |

---

## Nhóm 4 — Thanh toán & Hạn mức

| Cột | Kiểu | Mô tả | Ví dụ thực tế |
|-----|------|-------|---------------|
| **CREDIT_LIMIT** | float | **Hạn mức** thẻ tín dụng tối đa. | Hạn mức: $10,000 → ngân hàng tin tưởng khách |
| **PAYMENTS** | float | Tổng **số tiền** đã thanh toán (trả nợ). | Trả $2,000/tháng → khách có trách nhiệm tài chính |
| **MINIMUM_PAYMENTS** | float | Số tiền **than toán tối thiểu** bắt buộc mỗi tháng (thường ~5% số dư). | Minimum payment: $50/tháng |
| **PRC_FULL_PAYMENT** | float (0–1) | Tỷ lệ **thanh toán đủ** (trả hết số dư). 1 = luôn trả hết nợ. | 0.1 → chỉ trả đủ 10% số dư; 0.9 → thường xuyên trả hết |
| **TENURE** | int | **Thời hạn** sử dụng thẻ (tháng). Hầu hết = 12 (mới nhất). | TENURE = 12 → khách dùng thẻ từ đầu (thường là khách mới) |

---

## Kiểm thức giá trị (Từ mô tả dataset)

| Thống kê | Giá trị thường gặp |
|----------|--------------------|
| Số dư trung bình | ~$1,564 |
| Hạn mức trung bình | ~$4,494 |
| Thanh toán trung bình | ~$1,733 |
| Tỷ lệ trả đủ trung bình | ~15% (= 0.153) |
| Missing values | MINIMUM_PAYMENTS: 313, CREDIT_LIMIT: 1 |

---

## Cách dùng trong phân tích

Dataset này **không có target/label có sẵn** → dùng **unsupervised clustering (K-Means)** để phân khúc khách hàng thành 4 nhóm:

- **Cluster 0** - High Value Customers (số dư cao + mua nhiều)
- **Cluster 1** - Low Spending Customers (ít mua, số dư thấp)
- **Cluster 2** - Cash Advance Users (thường xuyên rút tiền mặt)
- **Cluster 3** - Potential Premium Customers (thanh toán tốt + hạn mức cao)