import torch
from transformers import AutoImageProcessor

from short_sport.video_trainer.classifier.timesformer import TimeSformer
from video_utils import sample_frames_indices


class InferenceEngine:
    """
    Lớp dùng để thực hiện suy luận (inference) trên dữ liệu video bằng các mô hình deep learning như TimeSformer.

    Tham số:
        model_name (str): Tên của mô hình pretrained (mặc định: 'timesformer-base-finetuned-k600').
        model_type (str): Loại mô hình sử dụng ('timesformer', 'videomae', 'videomaev2', 'uniform').
        n_frames (int): Số lượng khung hình (frame) được lấy mẫu từ video (mặc định: 16).
        n_classes (int): Số lượng nhãn đầu ra của mô hình (mặc định: 18).
        is_mutimodal (bool): Có sử dụng dữ liệu đa phương thức (multimodal) hay không (mặc định: False).
        checkpoint_path (str): Đường dẫn đến file checkpoint của mô hình (mặc định: None).
    """
    def __init__(self,
                 model_name: str = 'timesformer-base-finetuned-k600',
                 model_type: str = 'timesformer', 
                 n_frames: int= 16, 
                 n_classes: int = 18, 
                 is_mutimodal: bool = False,
                 checkpoint_path: str = None):
        
        super().__init__()
        self.model_name = model_name
        assert self.model_type in {'timesformer', 'uniform', 'videomae', 'videomaev2'}
        self.model_type = model_type
        
        assert n_frames > 0, "Number of frames must be positive"
        assert n_classes > 0, "Number of classes must be positive"
        self.n_frames = n_frames
        self.n_classes = n_classes
        self.checkpoint_path = checkpoint_path
        self.is_mutimodal = is_mutimodal
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not self.is_mutimodal:
            if self.model_type == 'timesformer':
                self.model = TimeSformer(n_frames=n_frames, num_classes=n_classes)
            elif self.model_type == 'videomae':
                self.model = VideoMae(n_frames=n_frames, num_classes=n_classes)
            elif self.model_type == 'videomaev2':
                self.model = VideoMaeV2(n_frames=n_frames, num_classes=n_classes)
            else:
                raise Exception(f'Unknown model type: {self.model_type}')
        if self.checkpoint_path:
            self.model = self._initialize_model_checkpoint(self.checkpoint_path)
    def frame_idx(self, video, method):
        """ 
        Example output format
        [<PIL.Image.Image image mode=RGB size=398x224>,
        <PIL.Image.Image image mode=RGB size=398x224>,
        .....
        <PIL.Image.Image image mode=RGB size=398x224>,
        <PIL.Image.Image image mode=RGB size=398x224>,
        """
        assert method in {'from_middle', 'from_top', 'from_bottom'}
        return sample_frames_indices(video, method, spacing=self.n_frames)
    
    def _initialize_video(self, video, method):
        video_lookup = self.frame_idx(video, method)
        inputs = self.image_processor(list(video_lookup), return_tensors="pt").to(self.device)
        return inputs

    def _initialize_model_checkpoint(self, checkpoint_path):
        assert self.model_type in checkpoint_path 
        ckpt = torch.load(checkpoint_path)
        optimizer = Adam([{"params": self.model.parameters(), "lr": 0.00001}])
        self.model.backbone.load_state_dict(ckpt["backbone"])
        self.model.classifier.load_state_dict(ckpt["classifier"])
        optimizer.load_state_dict(ckpt["optimizer"])
        return self.model
    @staticmethod
    def binary2label(values: torch.Tensor):
        return torch.nonzero(values[0], as_tuple=True)[0].tolist()
    
    def __call__(self, video, method='from_middle', threshold: float = 0.5):
        inputs=  self._initialize_video(video, method)
        self.model = self._initialize_model_checkpoint(self.checkpoint_path)
        
        with torch.no_grad():
            outputs = self.model.backbone(**inputs)
            logits = self.model.classifier(outputs[0][:, 0])
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= threshold).int()
        return InferenceEngine.binary2label(predictions)