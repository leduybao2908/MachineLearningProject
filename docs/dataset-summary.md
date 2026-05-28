# Dataset Summary & Conclusions from df.describe()

---

## 1. Thông tin chung

| Chỉ số | Giá trị |
|--------|---------|
| Tổng khách hàng | 8,950 |
| Missing values | MINIMUM_PAYMENTS: 313, CREDIT_LIMIT: 1 |

---

## 2. Nhóm 1 — Số dư (BALANCE)

| | Giá trị |
|---|---|
| Trung bình | **$1,564** |
| Trung vị (50%) | **$873** |
| Tứ phân vị 25% | **$128** |
| Tứ phân vị 75% | **$2,054** |
| Max | **$19,043** |

**Kết luận:**
- **Trung vị ($873) < Trung bình ($1,564)** → Phân phối **lệch phải** (right-skewed)
- → **Đa số khách** có số dư thấp, nhưng một **thiểu số khách** có số dư rất cao kéo trung bình lên
- ~50% khách có số dư dưới $873
- ~75% khách có số dư dưới $2,054

---

## 3. Nhóm 2 — Hành vi mua hàng

### Tổng quan số tiền

| | PURCHASES | ONEOFF_PURCHASES | INSTALLMENTS_PURCHASES |
|---|---|---|---|
| Trung bình | **$1,003** | **$592** | **$411** |
| Trung vị | **$361** | **$38** | **$89** |
| Max | **$49,039** | **$40,761** | **$22,500** |

**Kết luận:**
- `ONEOFF_PURCHASES` có trung bình cao hơn `INSTALLMENTS_PURCHASES` → Khách hàng **ưa mua đứt hơn mua trả góp**
- **Trung bình >> Trung vị** → Phân phối lệch phải mạnh, thiểu số mua rất nhiều kéo giá trị trung bình lên
- Tứ phân vị 25%: PURCHASES = $39 → **25% khách mua rất ít** (dưới $39)

### Tần suất mua hàng

| | PURCHASES_FREQUENCY | ONEOFF_PREQ_FREQUENCY | INSTALL_FREQUENCY |
|---|---|---|---|
| Trung bình | **0.49** | **0.20** | **0.36** |
| Tứ phân vị 25% | 0.08 | 0.00 | 0.00 |

**Kết luận:**
- PURCHASES_FREQUENCY TB = 0.49 → Khách mua **ít hơn nửa thời gian** (không active lắm)
- ONEOFF_FREQUENCY thấp (0.20) xác nhận khách ít mua đứt
- INSTALL_FREQUENCY (0.36) > ONEOFF_FREQUENCY (0.20) → Trả góp phổ biến hơn khi có giao dịch

### Số giao dịch mua

| | PURCHASES_TRX |
|---|---|
| Trung bình | **14.7 giao dịch** |
| Trung vị | **7 giao dịch** |
| Tứ phân vị 25% | **1 giao dịch** |

**Kết luận:**
- 25% khách chỉ mua **1 lần** trong 6 tháng
- Trung vị = 7 → **Nửa khách** có ít hơn 7 giao dịch
- Max = 358 → Khách active nhất mua ~1 lần/ngày

---

## 4. Nhóm 3 — Rút tiền mặt

| | CASH_ADVANCE | CASH_ADVANCE_FREQUENCY | CASH_ADVANCE_TRX |
|---|---|---|---|
| Trung bình | **$979** | **0.135** | **3.2 lần** |
| Trung vị | **$0** | **0** | **0** |

**Kết luận:**
- **Trung vị = 0** → **Hơn 50% khách không bao giờ rút tiền mặt**
- Nhưng một nhóm nhỏ rút **rất nhiều** (max = $47,137)
- CASH_ADVANCE_FREQUENCY TB = 0.135 → Hiếm khi rút, nhưng khi rút thì lặp lại

---

## 5. Nhóm 4 — Thanh toán & Hạn mức

### Hạn mức tín dụng

| | CREDIT_LIMIT |
|---|---|
| Trung bình | **$4,494** |
| Trung vị | **$3,000** |
| Tứ phân vị 75% | **$6,500** |

**Kết luận:**
- Hạn mức phân bố **lệch phải** (nhiều khách hạn mức thấp, ít khách hạn mức rất cao)
- 50% khách có hạn mức từ $50 - $3,000
- Max = $30,000 → Khách VIP có hạn mức cao gấp 10 lần trung bình

### Thanh toán

| | PAYMENTS | PRC_FULL_PAYMENT |
|---|---|---|
| Trung bình | **$1,733** | **0.154 (15.4%)** |
| Trung vị | **$857** | **0** |

**Kết luận:**
- **Trung vị PRC_FULL_PAYMENT = 0** → **Hơn 50% khách KHÔNG trả hết nợ** (toàn bộ số dư)
- Trung bình PRC_FULL_PAYMENT = 15.4% → Chỉ ~15% số lần khách trả đủ
- PAYMENTS TB ($1,733) > BALANCE TB ($1,564) → Nhìn chung khách có xu hướng **trả nhiều hơn số nợ**

### Thời hạn sử dụng & Missing Payments

| | TENURE | MINIMUM_PAYMENTS |
|---|---|---|
| Trung bình | **11.5 tháng** | **$864** |
| Trung vị | **12 tháng** | **$312** |

**Kết luận:**
- TENURE trung bình 11.5/12 tháng → Hầu hết khách **dùng thẻ từ đầu** (khách mới)
- MINIMUM_PAYMENTS median < mean → Lệch phải (vài khách trả rất nhiều)
- 313 missing MINIMUM_PAYMENTS → Nhóm khách không có khoản thanh toán tối thiểu (có thể trả đủ → không có minimum)

---

## 6. Tổng kết — 4 Insight quan trọng

### Insight 1: Đa số khách hàng nhỏ, ít hoạt động
- 25% chỉ mua 1 lần, 50% mua ≤7 lần trong 6 tháng
- 25% có số dư dưới $128 → Ít chi tiêu

### Insight 2: Phân hóa mạnh giữa các nhóm khách
- Số dư trung bình gấp đôi trung vị → Thiểu số "heavy user" kéo trung bình lên
- Hạn mức max gấp 6 lần trung bình → Khách hàng rất đa dạng về tài chính

### Insight 3: Rủi ro tài chính tập trung ở nhóm nhỏ
- >50% không trả hết nợ (PRC_FULL_PAYMENT median = 0)
- Nhóm rút tiền mặt (Cash Advance) có hành vi rủi ro

### Insight 4: Hành vi trả góp phổ biến hơn mua đứt
- INSTALLMENTS_PURCHASES_FREQUENCY (0.36) > ONEOFF_PURCHASES_FREQUENCY (0.20)
- Nhưng khi mua đứt, số tiền trung bình cao hơn ($592 vs $411)

---

## 7. Gợi ý cho Clustering

Các feature có **phân phối méo nhất** (good for clustering):

1. **CASH_ADVANCE** — median = 0, max = $47K → Tách được nhóm "nghiện rút tiền"
2. **ONEOFF_PURCHASES** — median = $38, max = $40K → Tách VIP buyer
3. **PRC_FULL_PAYMENT** — median = 0 → Tách nhóm trả đủ vs không trả đủ
4. **PURCHASES_FREQUENCY** — max = 1.0 → Tách khách siêu active