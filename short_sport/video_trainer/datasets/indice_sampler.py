import numpy

def random_sampling(num_frames, total_length, random_shift: bool = False):
    if num_frames <= total_length:
        indices = numpy.linspace(0, num_frames - 1, total_length, dtype=int)
    else:
        ticks = numpy.linspace(0, num_frames, total_length + 1, dtype=int)
        if random_shift:
            indices = ticks[:-1] + numpy.random.randint(ticks[1:] - ticks[:-1])
        else:
            indices = ticks[:-1] + (ticks[1:] - ticks[:-1]) // 2
    return indices

def incremental_sampling(n: int, total_values: int, method: str = "noise+"):
    """
    Chọn n giá trị duy nhất theo phân phối tăng dần (incremental) 
    với cải tiến để tránh luôn chọn các index đầu tiên.

    Args:
        n (int): Số lượng index cần chọn (n <= total_values).
        total_values (int): Tổng số index có thể chọn.
        temperature (float): Điều chỉnh độ sắc nét.
        method (str): Phương pháp điều chỉnh ["noise+", "shift+", "gaussian+"].

    Returns:
        selected_indices (list): Danh sách n index được chọn.
    """
    temperature=5.0
    indices = numpy.arange(total_values)
    
    if method == "noise+":
        # Tăng nhiễu và giảm xác suất cho các index đầu
        start_index = max(2, int(total_values * 0.1))  # Bắt đầu từ index thứ 2 hoặc 10%
        logits = numpy.zeros(total_values)
        logits[start_index:] = numpy.exp((indices[start_index:] - start_index) / temperature)
        logits += numpy.random.uniform(0.01, 0.05, size=total_values)
        
    elif method == "shift+":
        # Dịch chuyển và giảm xác suất cho các index đầu
        start_index = max(3, int(total_values * 0.15))  # Bắt đầu từ index thứ 3 hoặc 15%
        shift_factor = 3
        logits = numpy.zeros(total_values)
        logits[start_index:] = numpy.exp((indices[start_index:] - start_index + shift_factor) / temperature)
        logits += 0.01 * numpy.linspace(0, 1, total_values)
        
    elif method == "gaussian+":
        # Gaussian với peak ở phần sau và giảm xác suất cho các index đầu
        start_index = max(4, int(total_values * 0.2))  # Bắt đầu từ index thứ 4 hoặc 20%
        mean_index = total_values * 0.7  # Peak ở 70% vị trí
        std_dev = total_values / 4  # Điều chỉnh độ rộng
        
        logits = numpy.zeros(total_values)
        logits[start_index:] = numpy.exp(-((indices[start_index:] - mean_index) ** 2) / (2 * std_dev ** 2))
        logits += 0.01  # Thêm một lượng noise nhỏ
    
    else:
        raise ValueError("Phương pháp không hợp lệ. Chọn giữa 'noise+', 'shift+' hoặc 'gaussian+'.")

    # Chuẩn hóa bằng softmax
    probs = logits / numpy.sum(logits)
    
    # Chọn n giá trị duy nhất theo xác suất
    selected_indices = numpy.random.choice(indices, size=n, replace=False, p=probs)
    selected_indices = [int(i) for i in selected_indices]
    return sorted(selected_indices)

def decremental_sampling(n_frames: int, total_values: int, method: str = "noise+"):
    """
    Chọn n giá trị duy nhất theo phân phối giảm dần từ cuối, 
    ưu tiên các index cuối của video.

    Args:
        n (int): Số lượng index cần chọn (n <= total_values).
        total_values (int): Tổng số index có thể chọn.
        temperature (float): Điều chỉnh độ sắc nét.
        method (str): Phương pháp điều chỉnh ["noise+", "shift+", "gaussian+"].

    Returns:
        selected_indices (list): Danh sách n index được chọn.
    """
    temperature=5.0
    indices = numpy.arange(total_values)
    
    # Đảo ngược indices để tính toán từ cuối
    reversed_indices = total_values - 1 - indices
    
    if method == "noise+":
        # Tăng xác suất cho các index cuối với nhiễu nhỏ
        end_start_index = max(2, int(total_values * 0.6))  # Bắt đầu từ 60% video
        logits = numpy.zeros(total_values)
        logits[end_start_index:] = numpy.exp((reversed_indices[end_start_index:]) / temperature)
        logits += numpy.random.uniform(0.01, 0.05, size=total_values)
        logits[end_start_index:] *= 1.5  # Tăng thêm trọng số cho các index cuối
        
    elif method == "shift+":
        # Dịch chuyển và tăng xác suất cho các index cuối
        end_start_index = max(3, int(total_values * 0.7))  # Bắt đầu từ 70% video
        shift_factor = 3
        logits = numpy.zeros(total_values)
        logits[end_start_index:] = numpy.exp((reversed_indices[end_start_index:] + shift_factor) / temperature)
        logits += 0.01 * numpy.linspace(0, 1, total_values)
        logits[end_start_index:] *= 2  # Tăng gấp đôi trọng số cho các index cuối
        
    elif method == "gaussian+":
        # Gaussian với peak ở phần cuối video
        end_start_index = max(4, int(total_values * 0.8))  # Bắt đầu từ 80% video
        mean_index = total_values * 0.9  # Peak gần cuối video
        std_dev = total_values / 4  # Điều chỉnh độ rộng
        
        logits = numpy.zeros(total_values)
        logits[end_start_index:] = numpy.exp(-((indices[end_start_index:] - mean_index) ** 2) / (2 * std_dev ** 2))
        logits += 0.01  # Thêm một lượng noise nhỏ
        logits[end_start_index:] *= 2.5  # Tăng trọng số cho các index cuối
    
    else:
        raise ValueError("Phương pháp không hợp lệ. Chọn giữa 'noise+', 'shift+' hoặc 'gaussian+'.")

    # Chuẩn hóa bằng softmax
    probs = logits / numpy.sum(logits)
    
    # Chọn n giá trị duy nhất theo xác suất
    selected_indices = numpy.random.choice(indices, size=n_frames, replace=False, p=probs)
    selected_indices = [int(i) for i in selected_indices]
    return sorted(selected_indices)
    
def twopeak_sampling(n_frames: int, total_values: int, method: str = "concentrated"):
    """
    Chọn n giá trị với phân phối hai đỉnh rõ ràng và tách biệt.

    Args:
        n (int): Số lượng index cần chọn (n <= total_values).
        total_values (int): Tổng số index có thể chọn.
        temperature (float): Điều chỉnh độ sắc nét (giá trị nhỏ = tập trung hơn).
        method (str): Phương pháp chọn lọc.

    Returns:
        selected_indices (list): Danh sách n index được chọn.
    """
    temperature = 5.0
    indices = numpy.arange(total_values)
    
    # Khởi tạo logits với giá trị rất nhỏ
    logits = numpy.ones(total_values) * 0.0001
    
    if method == "distinct":
        # Hai đỉnh rõ ràng, cách xa nhau
        peak1 = int(total_values * 0.25)  # Đỉnh thứ nhất ở 25%
        peak2 = int(total_values * 0.75)  # Đỉnh thứ hai ở 75%
        
        # Tạo hai đỉnh riêng biệt bằng cách gán giá trị trực tiếp
        peak1_width = 2  # Độ rộng đỉnh 1
        peak2_width = 2  # Độ rộng đỉnh 2
        
        # Gán giá trị cao cho các vị trí gần đỉnh
        for i in range(peak1 - peak1_width, peak1 + peak1_width + 1):
            if 0 <= i < total_values:
                distance = abs(i - peak1)
                logits[i] = 10.0 * (1 - distance / (peak1_width + 1))
                
        for i in range(peak2 - peak2_width, peak2 + peak2_width + 1):
            if 0 <= i < total_values:
                distance = abs(i - peak2)
                logits[i] = 10.0 * (1 - distance / (peak2_width + 1))
        
    elif method == "bimodal":
        # Hai đỉnh với phân phối bimodal rõ ràng
        peak1 = int(total_values * 0.2)  # Đỉnh thứ nhất ở 20%
        peak2 = int(total_values * 0.8)  # Đỉnh thứ hai ở 80%
        
        # Tạo giá trị cao cho các đỉnh và giảm dần
        peak1_width = 3  # Độ rộng đỉnh 1
        peak2_width = 3  # Độ rộng đỉnh 2
        
        # Tạo phân phối bimodal với đỉnh rõ ràng
        for i in range(total_values):
            dist1 = abs(i - peak1)
            dist2 = abs(i - peak2)
            
            if dist1 <= peak1_width:
                logits[i] = max(logits[i], 15.0 * (1 - dist1 / (peak1_width + 1)))
            
            if dist2 <= peak2_width:
                logits[i] = max(logits[i], 15.0 * (1 - dist2 / (peak2_width + 1)))
                
        # Giảm mạnh xác suất vùng giữa
        mid_point = (peak1 + peak2) // 2
        mid_width = (peak2 - peak1) // 3
        for i in range(mid_point - mid_width, mid_point + mid_width):
            if 0 <= i < total_values:
                logits[i] = 0.0001  # Gần như bằng 0
        
    elif method == "concentrated":
        # Hai đỉnh tập trung cao với vùng trũng giữa
        peak1 = int(total_values * 0.3)  # Đỉnh thứ nhất ở 30%
        peak2 = int(total_values * 0.7)  # Đỉnh thứ hai ở 70%
        
        # Tạo phân phối với đỉnh rất cao và vùng trũng rõ ràng
        for i in range(total_values):
            # Tính khoảng cách tới đỉnh gần nhất
            dist_to_peak = min(abs(i - peak1), abs(i - peak2))
            
            if dist_to_peak <= 1:  # Chỉ đỉnh và điểm liền kề
                logits[i] = 20.0
            elif dist_to_peak <= 3:  # Vùng gần đỉnh
                logits[i] = 5.0
            elif dist_to_peak <= 5:  # Vùng xa hơn
                logits[i] = 0.5
            else:  # Vùng xa
                logits[i] = 0.0001
    
    else:
        raise ValueError("Phương pháp không hợp lệ.")

    # Áp dụng temperature để điều chỉnh độ sắc nét
    logits = numpy.power(logits, 1/temperature)
    
    # Chuẩn hóa bằng softmax
    probs = logits / numpy.sum(logits)
    
    # Chọn n giá trị duy nhất theo xác suất
    selected_indices = numpy.random.choice(indices, size=n_frames, replace=True, p=probs)
    selected_indices = [int(i) for i in selected_indices]
    return sorted(selected_indices)

