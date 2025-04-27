# Đăng ký task
TASK_REGISTRY = {}

def register_task(name):
    """Decorator để đăng ký task."""
    def decorator(func):
        TASK_REGISTRY[name] = func
        return func
    return decorator

def task(task_name, **kwargs):
    """Chạy task dựa trên tên, raise lỗi nếu task không hợp lệ."""
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Task '{task_name}' không hợp lệ. Các task hợp lệ: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_name](**kwargs)
