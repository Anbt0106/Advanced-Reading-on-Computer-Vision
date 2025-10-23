# BÁO CÁO PHÂN TÍCH THUẬT TOÁN DoG + HARRIS KEYPOINT DETECTION

## 1. TỔNG QUAN THUẬT TOÁN

Chương trình `dog_harris_detections.ipynb` triển khai thuật toán phát hiện điểm quan trọng (keypoint) kết hợp hai phương pháp:
- **Difference of Gaussian (DoG)**: Để tìm điểm bất biến theo tỷ lệ (scale-invariant)
- **Harris Corner Response**: Để nhấn mạnh các điểm góc mạnh

## 2. PHÂN TÍCH CHI TIẾT CÁC THÀNH PHẦN

### 2.1. Khởi tạo và Tham số (Dòng 16-31)
```python
SIGMA0 = 1.6          # Độ lệch chuẩn Gaussian cơ sở
SCALES_PER_OCT = 3    # Số scale trong mỗi octave
NUM_OCTAVES = 4       # Số octave (tầng tỷ lệ)
CONTRAST_TH = 0.03    # Ngưỡng độ tương phản
EDGE_R = 10           # Tỷ số ngưỡng loại bỏ edge
HARRIS_K = 0.04       # Hệ số Harris (thường 0.04-0.06)
HARRIS_REL_TH = 0.01  # Ngưỡng tương đối Harris
NMS_RADIUS = 3        # Bán kính Non-Maximum Suppression
```

**Tác dụng**: Định nghĩa các tham số điều khiển độ nhạy và chất lượng của thuật toán phát hiện keypoint.

### 2.2. Xây dựng Gaussian Pyramid (Dòng 37-56)
```python
def build_gaussian_pyramid(gray, num_octaves=NUM_OCTAVES, s=SCALES_PER_OCT, sigma0=SIGMA0):
```

**Mục đích lý thuyết**: Tạo không gian tỷ lệ (scale space) để phát hiện đặc trưng ở nhiều mức độ chi tiết khác nhau.

**Cách triển khai**:
- Tạo kim tự tháp Gaussian với 4 octave, mỗi octave có 6 ảnh (3 + 3 thêm)
- Sử dụng hệ số k = 2^(1/s) để tính sigma cho mỗi mức
- Octave đầu: làm mờ ảnh gốc với sigma phù hợp
- Các octave tiếp theo: downsampling từ octave trước với tỷ lệ 1:2

**So sánh với lý thuyết**: Đúng với thuật toán SIFT gốc, tạo không gian tỷ lệ liên tục để đảm bảo tính bất biến tỷ lệ.

### 2.3. Xây dựng DoG Pyramid (Dòng 58-63)
```python
def build_dog_pyramid(gauss_pyr):
    dog_pyr = []
    for octave_imgs in gauss_pyr:
        dogs = [octave_imgs[i] - octave_imgs[i-1] for i in range(1, len(octave_imgs))]
```

**Mục đích lý thuyết**: DoG xấp xỉ Laplacian of Gaussian (LoG), phát hiện blob structures và điểm cực trị.

**Cách triển khai**: Tính hiệu của các cặp ảnh Gaussian liên tiếp trong mỗi octave.

**So sánh với lý thuyết**: Chính xác theo công thức DoG = G(σ₂) - G(σ₁), là xấp xỉ hiệu quả của LoG.

### 2.4. Phát hiện Cực trị Cục bộ (Dòng 65-71)
```python
def is_local_extrema(cube):
    c = cube[1,1,1]
    if c > 0:
        return c == cube.max()
    else:
        return c == cube.min()
```

**Mục đích lý thuyết**: Tìm điểm cực đại/cực tiểu trong không gian 3D (x, y, scale).

**Cách triển khai**: Kiểm tra 26 điểm lân cận (3x3x3 cube minus điểm trung tâm).

**So sánh với lý thuyết**: Đúng với SIFT, đảm bảo điểm keypoint là cực trị trong cả không gian và tỷ lệ.

### 2.5. Loại bỏ Edge Response (Dòng 73-85)
```python
def pass_edge_response(dog, y, x, r=EDGE_R):
    Dxx = dog[y, x+1] + dog[y, x-1] - 2*dog[y, x]
    Dyy = dog[y+1, x] + dog[y-1, x] - 2*dog[y, x]
    Dxy = (dog[y+1, x+1] - dog[y+1, x-1] - dog[y-1, x+1] + dog[y-1, x-1]) * 0.25
```

**Mục đích lý thuyết**: Loại bỏ các điểm keypoint nằm trên edge, chỉ giữ lại corner points.

**Cách triển khai**: 
- Tính ma trận Hessian 2x2
- Sử dụng tỷ số eigenvalue để phân biệt corner và edge
- Điều kiện: (trace²/det) < ((r+1)²/r)

**So sánh với lý thuyết**: Đúng với phương pháp Harris-Laplace, sử dụng tỷ số eigenvalue để loại bỏ edge.

### 2.6. Tính Harris Response (Dòng 87-98)
```python
def harris_response(img, k=HARRIS_K, win_sigma=1.0):
    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    # ... tính ma trận structure tensor
    R = detM - k*(traceM**2)
```

**Mục đích lý thuyết**: Đo "cornerness" - mức độ góc của điểm, bổ sung cho DoG detector.

**Cách triển khai**:
- Tính gradient Ix, Iy bằng Sobel
- Tạo structure tensor từ Ixx, Iyy, Ixy
- Áp dụng Gaussian smoothing
- Tính Harris response: R = det(M) - k×trace(M)²

**So sánh với lý thuyết**: Chính xác theo công thức Harris corner detector gốc.

### 2.7. Non-Maximum Suppression (Dòng 100-115)
```python
def nonmax_suppression(points, radius):
    order = np.argsort(-pts[:,2])  # Sắp xếp theo score giảm dần
    # Loại bỏ các điểm gần nhau trong bán kính cho trước
```

**Mục đích lý thuyết**: Loại bỏ các keypoint dư thừa, chỉ giữ lại điểm mạnh nhất trong vùng lân cận.

**Cách triển khai**: Greedy algorithm - chọn điểm có score cao nhất, loại bỏ các điểm lân cận.

### 2.8. Hàm Chính - Phát hiện Keypoint (Dòng 117-175)
```python
def detect_keypoints_dog_harris(gray, ...):
```

**Quy trình tổng hợp**:
1. **Xây dựng pyramid**: Gaussian và DoG pyramid
2. **Tính Harris response**: Cho mỗi mức Gaussian
3. **Tìm candidate**: Điểm cực trị trong DoG space
4. **Lọc keypoint**: 
   - Kiểm tra contrast threshold
   - Kiểm tra edge response
   - Kiểm tra Harris response threshold
5. **Non-Maximum Suppression**: Loại bỏ điểm dư thừa
6. **Mapping về ảnh gốc**: Chuyển đổi tọa độ về kích thước ban đầu

## 3. SO SÁNH VỚI LÝ THUYẾT

### 3.1. Điểm Mạnh
- **Kết hợp DoG + Harris**: Tận dụng ưu điểm của cả hai - scale invariance của DoG và corner detection của Harris
- **Triển khai đúng chuẩn**: Theo đúng các bước của SIFT và Harris detector
- **Filtering hiệu quả**: Nhiều tầng lọc để đảm bảo chất lượng keypoint

### 3.2. Điểm Cải tiến so với SIFT thuần túy
- **Thêm Harris response**: Tăng cường khả năng phát hiện corner
- **Dual filtering**: Sử dụng cả DoG contrast và Harris threshold
- **Visualization**: Xuất Harris heatmaps để phân tích

### 3.3. Ứng dụng thực tế
- **Robust keypoint detection**: Phù hợp cho feature matching, object recognition
- **Scale invariant**: Hoạt động tốt với ảnh ở nhiều tỷ lệ khác nhau
- **Corner emphasis**: Tập trung vào các góc có ý nghĩa hình học

## 4. KẾT LUẬN

Chương trình triển khai thành công thuật toán hybrid DoG-Harris với:
- **Tính chính xác cao** trong việc phát hiện keypoint
- **Kết hợp thông minh** giữa scale-space theory (DoG) và corner detection (Harris)
- **Triển khai đầy đủ** các bước lọc và tinh chế keypoint
- **Khả năng visualization** tốt để phân tích kết quả

Thuật toán này đặc biệt hiệu quả cho các ứng dụng computer vision cần keypoint ổn định và có ý nghĩa hình học cao.
