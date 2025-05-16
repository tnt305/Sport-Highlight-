from .timesformer_base_v1 import TimeSformerExecutor
from .videomae import VideoMaeExecutor
from .timesformer_clip import TimeSformerCLIPInitExecutor
from .timesformer_gnn import TimeSformerGNNExecutor
from .timesformer_epoch_based import TimeSformerExecutor, TimesformerBase
from .timesformer_epoch_based_with_augment import TimeSformerEpochBasedWithAugmentationExecutor
from .timesformer_epch_w_aug_w_videorope import TimeSformerEpochBasedWithAugmentationRoPEExecutor
from .video_audio_fusion import VAFusionExecuter
from .video_audio_fusion_v2 import VAHarderFusionExecuter
from .uniformer_va_fusion import UniformerVAFusionExecuter
# from .timesformer_base_v1 import TimeSformerExecutor

# MODEL_REGISTER = {
#     "video": TimesformerBase,
#     "video_roformer": TimeSformerEpochBasedWithAugmentationRoPEExecutor,
#     "video_audio": AudioVideoFusionClassifier,
#     "video_text": TextVideoFusionClassifier,
# }