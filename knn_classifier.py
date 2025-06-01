# Lớp KNNClassifier để thực hiện thuật toán K-Nearest Neighbors
# Dùng để nhận diện chữ số từ tập MNIST

# Import các thư viện cần thiết
import numpy as np  # Thư viện xử lý mảng số học
from scipy import stats  # Thư viện tính toán thống kê (dùng stats.mode để tìm nhãn phổ biến)
from scipy.spatial import KDTree  # Cấu trúc dữ liệu KD-Tree để tìm kiếm hàng xóm gần nhất

# Lớp KNNClassifier
class KNNClassifier:
    def __init__(self, k: int = 3, use_kdtree: bool = True, weighted_voting: bool = False):
        """
        Khởi tạo mô hình KNNClassifier.
        Args:
            k (int): Số lượng hàng xóm gần nhất để xét (mặc định: 3)
            use_kdtree (bool): Có dùng KD-Tree để tìm kiếm hàng xóm không (mặc định: True)
            weighted_voting (bool): Có dùng trọng số khoảng cách khi voting không (mặc định: False, chưa triển khai)
        """
        self.k = k  # Lưu số lượng hàng xóm gần nhất
        self.use_kdtree = use_kdtree  # Lưu lựa chọn dùng KD-Tree
        self.weighted_voting = weighted_voting  # Lưu lựa chọn voting có trọng số
        self.X_train = None  # Lưu dữ liệu huấn luyện (ban đầu là None)
        self.y_train = None  # Lưu nhãn huấn luyện (ban đầu là None)
        self.kdtree = None  # Lưu đối tượng KD-Tree (ban đầu là None)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Huấn luyện mô hình KNN bằng cách lưu dữ liệu huấn luyện và xây dựng KD-Tree (nếu dùng).
        Args:
            X (np.ndarray): Ma trận dữ liệu huấn luyện (shape: [số mẫu, số đặc trưng])
            y (np.ndarray): Mảng nhãn huấn luyện (shape: [số mẫu])
        """
        # Kiểm tra số mẫu của X và y có khớp nhau không
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        self.X_train = np.array(X)  # Lưu dữ liệu huấn luyện
        self.y_train = np.array(y)  # Lưu nhãn huấn luyện
        # Nếu dùng KD-Tree, xây dựng cấu trúc từ X_train
        if self.use_kdtree:
            self.kdtree = KDTree(self.X_train)

    def _predict_one(self, test_point: np.ndarray) -> int:
        """
        Dự đoán nhãn cho một điểm thử nghiệm.
        Args:
            test_point (np.ndarray): Điểm thử nghiệm (shape: [số đặc trưng])
        Returns:
            int: Nhãn dự đoán
        """
        # Tìm k hàng xóm gần nhất
        if self.use_kdtree and self.kdtree is not None:
            # Nếu dùng KD-Tree, tìm k hàng xóm nhanh chóng
            distances, indices = self.kdtree.query(test_point, k=self.k)
        else:
            # Nếu không dùng KD-Tree, tính khoảng cách Euclidean thủ công
            distances = np.sqrt(np.sum((self.X_train - test_point) ** 2, axis=1))
            indices = np.argpartition(distances, self.k)[:self.k]  # Lấy chỉ số k điểm gần nhất
            distances = distances[indices]  # Lấy khoảng cách tương ứng
        k_labels = self.y_train[indices]  # Lấy nhãn của k hàng xóm
        # Tìm nhãn phổ biến nhất trong k hàng xóm
        return stats.mode(k_labels, keepdims=False)[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán nhãn cho tập hợp điểm thử nghiệm.
        Args:
            X (np.ndarray): Ma trận điểm thử nghiệm (shape: [số mẫu thử nghiệm, số đặc trưng])
        Returns:
            np.ndarray: Mảng nhãn dự đoán (shape: [số mẫu thử nghiệm])
        """
        X = np.array(X)  # Chuyển dữ liệu thử nghiệm thành mảng numpy
        # Kiểm tra xem mô hình đã được huấn luyện chưa
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model must be fitted before prediction.")
        # Kiểm tra số đặc trưng của X có khớp với X_train không
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError("Test data must have same number of features as training data.")
        # Dự đoán nhãn cho từng điểm thử nghiệm
        predictions = [self._predict_one(test_point) for test_point in X]
        return np.array(predictions)  # Trả về mảng nhãn dự đoán