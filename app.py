# Giao diện Streamlit để nhận diện chữ số MNIST
# Hỗ trợ upload ảnh, tùy chọn chuyển về nền đen chữ trắng, chọn k, hiển thị ảnh ban đầu, ảnh xử lý, và k hàng xóm gần nhất
import streamlit as st
import numpy as np
from PIL import Image
from knn_classifier import KNNClassifier

# Tải dữ liệu huấn luyện từ .npy
@st.cache_resource
def load_data():
    X_train = np.load('X_train_mnist.npy')
    y_train = np.load('y_train_mnist.npy')
    return X_train, y_train

X_train, y_train = load_data()

# Giao diện Streamlit
st.title("Nhận Diện Chữ Số MNIST")
st.write("Tải lên ảnh chữ số (28x28) để dự đoán.")

# Selectbox để chọn k
k = st.selectbox("Chọn giá trị k:", [3, 5, 7, 9], index=0)

# Tải và huấn luyện mô hình KNN với k được chọn
@st.cache_resource(hash_funcs={KNNClassifier: lambda _: k})
def load_knn_model(k_value):
    knn = KNNClassifier(k=k_value, use_kdtree=True, weighted_voting=False)
    knn.fit(X_train, y_train)
    return knn

knn = load_knn_model(k)

# Checkbox để bật/tắt chuyển về nền đen, chữ trắng
to_black_background = st.checkbox("Chuyển về nền đen, chữ trắng", value=False)

# Hàm căn giữa chữ số trong ảnh (giữ nguyên)
def center_digit_in_image(pixels, threshold=0.5):
    """
    Căn giữa chữ số trong ảnh 28x28 bằng cách tính trọng tâm và dịch chuyển.
    Args:
        pixels: Ảnh (numpy array) đã chuẩn hóa về [0.0, 1.0].
        threshold: Ngưỡng để phân biệt chữ số và nền.
    Returns:
        Ảnh đã được căn giữa (numpy array).
    """
    binary_image = np.where(pixels >= threshold, 1.0, 0.0)
    rows, cols = np.where(binary_image == 1.0)
    if len(rows) == 0 or len(cols) == 0:
        return pixels

    centroid_y = np.mean(rows)
    centroid_x = np.mean(cols)
    center_y, center_x = 14, 14
    shift_y = int(round(center_y - centroid_y))
    shift_x = int(round(center_x - centroid_x))
    centered_image = np.zeros_like(pixels)
    
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            new_i = i + shift_y
            new_j = j + shift_x
            if 0 <= new_i < 28 and 0 <= new_j < 28:
                centered_image[new_i, new_j] = pixels[i, j]

    return centered_image

# Hàm dự đoán và lấy hàng xóm gần nhất (giữ nguyên)
def predict_with_neighbors(test_point, knn):
    """
    Dự đoán nhãn và lấy chỉ số k hàng xóm gần nhất.
    """
    distances, indices = knn.kdtree.query(test_point.reshape(1, -1), k=knn.k)
    k_labels = y_train[indices[0]]
    prediction = np.bincount(k_labels).argmax()
    return prediction, indices[0]

# Upload ảnh với nhiều định dạng
uploaded_file = st.file_uploader("Chọn ảnh chữ số (PNG, JPG, JPEG, BMP, GIF, 28x28):", type=["png", "jpg", "jpeg", "bmp", "gif"])
if uploaded_file is not None:
    try:
        # Mở ảnh và chuyển đổi sang grayscale
        image = Image.open(uploaded_file)
        
        # Nếu ảnh có kênh alpha (RGBA) hoặc nhiều kênh màu (RGB), chuyển về grayscale (L)
        if image.mode in ('RGBA', 'RGB'):
            image = image.convert('L')
        elif image.mode != 'L':
            # Nếu ảnh không phải RGB/RGBA hoặc grayscale, thử chuyển đổi
            image = image.convert('L')

        # Resize ảnh về 28x28
        image = image.resize((28, 28))
        pixels = np.array(image, dtype=float)

        # Chuẩn hóa ảnh ban đầu về [0.0, 1.0] để hiển thị
        display_pixels_initial = pixels / 255.0

        # Chuẩn hóa ảnh để xử lý
        pixels = pixels / 255.0

        # Bước 1: Chuyển về nền đen, chữ trắng nếu người dùng chọn
        if to_black_background:
            threshold = 0.5
            if pixels.mean() > threshold:
                pixels = 1.0 - pixels
            pixels = np.where(pixels < threshold, 0.0, 1.0)

        # Bước 2: Căn giữa chữ số trong ảnh
        pixels = center_digit_in_image(pixels, threshold=0.5)

        # Hiển thị ảnh ban đầu và ảnh xử lý cạnh nhau (side by side) với kích thước ngang bằng
        display_pixels_processed = pixels
        col1, col2 = st.columns([1, 1])  # Đảm bảo hai cột có chiều rộng bằng nhau
        with col1:
            st.image(display_pixels_initial, caption="Ảnh ban đầu", use_column_width=True)
        with col2:
            st.image(display_pixels_processed, caption="Ảnh đã xử lý (căn giữa)", use_column_width=True)

        pixels_processed = pixels.flatten()

        if pixels_processed.shape != (784,):
            st.error(f"Kích thước pixel không hợp lệ: {pixels_processed.shape}, kỳ vọng (784,)")
        else:
            prediction, neighbor_indices = predict_with_neighbors(pixels_processed, knn)
            st.write(f"**Chữ số dự đoán**: {int(prediction)}")

            # Hiển thị k hàng xóm gần nhất
            st.write(f"**Các hàng xóm gần nhất (k={k}):**")
            cols = st.columns(k)
            for i, idx in enumerate(neighbor_indices):
                with cols[i]:
                    neighbor_image = X_train[idx].reshape(28, 28)
                    st.image(neighbor_image, caption=f"Hàng xóm {i+1}: Nhãn {int(y_train[idx])}", width=100)
    
    except Exception as e:
        st.error(f"Lỗi: {str(e)}")