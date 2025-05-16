from short_sport.video_trainer.datasets.transform import *
from torchvision.transforms import RandAugment
# from RandAugment import RandAugment

class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

def get_augmentation(mode='train'):
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    input_size = 224
    scale_size = 256
    if mode == 'train':
        unique = T.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                            GroupRandomHorizontalFlip(True),
                            GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                            GroupRandomGrayscale(p=0.2),
                            GroupGaussianBlur(p=0.0),
                            GroupSolarization(p=0.0)]
                            )
    else:
        unique = T.Compose([GroupScale(scale_size),
                            GroupCenterCrop(input_size)])

    common = T.Compose([Stack(roll=False),
                        ToTorchFormatTensor(div=True),
                        GroupNormalize(input_mean, input_std)])
    return T.Compose([unique, common])

def randAugment(transform_train,config):
    print('Using RandAugment!')
    transform_train.transforms.insert(0, GroupTransform(RandAugment(config.data.randaug.N, config.data.randaug.M)))
    return transform_train

class MixUpAugmentation:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, batch_data, batch_labels):
        """
        Áp dụng Mixup cho batch video
        batch_data: Tensor [batch_size, channels, frames, height, width]
        batch_labels: Tensor [batch_size, num_classes] 
        """
        batch_size = batch_data.size(0)
        indices = torch.randperm(batch_size)
        
        # Tạo hệ số mixup từ phân phối Beta
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mixup data
        mixed_data = lam * batch_data + (1 - lam) * batch_data[indices]
        
        # Mixup labels (cho multilabel)
        mixed_labels = lam * batch_labels + (1 - lam) * batch_labels[indices]
        
        return mixed_data, mixed_labels


class MultimodalMixupAugmentation:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, video, audio, labels):
        """
        Apply mixup to video, audio, and labels
        
        Args:
            video: Tensor of shape [batch_size, ...]
            audio: List of audio paths or features
            labels: Tensor of shape [batch_size, ...]
            
        Returns:
            mixed_video, mixed_audio, mixed_labels
        """
        batch_size = video.size(0)
        indices = torch.randperm(batch_size)
        
        # Sample lambda from beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mixup video
        mixed_video = lam * video + (1 - lam) * video[indices]
        
        # Mixup audio (handling paths)
        mixed_audio = []
        for i in range(batch_size):
            if np.random.random() < lam:
                mixed_audio.append(audio[i])
            else:
                mixed_audio.append(audio[indices[i]])
                
        # Mixup labels
        mixed_labels = lam * labels + (1 - lam) * labels[indices]
        
        return mixed_video, mixed_audio, mixed_labels
