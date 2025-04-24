import cv2
import pandas
import numpy
from tqdm import tqdm
from utils.dataset_level import read_csv
from src.color_domain.base import BaseColor
from sklearn.cluster import KMeans
from collections import Counter
from typing import List, Tuple, Optional, ClassVar
from pydantic import BaseModel, Field, validator


class BaseColor(BaseModel):
    """Base class for color processing operations."""
    STRATEGIES: ClassVar[List[str]] = ['centric', 'kmeans', 'hierarchy'] 
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-pydantic compatible types
    def get_color_name(self, r: int, g: int, b: int, color_data: pandas.DataFrame) -> str:
        """Find the closest matching color name for given RGB values."""
        raise NotImplementedError
    
    def get_color_main(self, image):
        raise NotImplementedError
    
class DominantColorDetector(BaseColor):
    strategy: str = Field(default='kmeans')
    
    n_colors: int = Field(
        default = 3,
        description="Number of color clusters for K-means"
    )
    _kmeans: Optional[KMeans] = None
    def __init__(self, **data):
        super().__init__(**data)
        if self.strategy == 'kmeans':
            self._init_kmeans()
    
    def _init_kmeans(self):
        """Initialize KMeans with optimized parameters."""
        self._kmeans = KMeans(
            random_state = 0,
            n_clusters=self.n_colors,
            init='k-means++'
        )
        
    @validator('strategy')
    def validate_strategy(cls, v):
        if v not in BaseColor.STRATEGIES:
            raise ValueError(f"Strategy must be one of the following: {BaseColor.STRATEGIES}")
        return v
    
    @staticmethod
    def hicherachy_color(img, level: int = 2) -> Tuple[int, int, int]:
        h, w = img.shape[:2]
        center_x, center_y = w // level, h // level
        b, g, r = img[center_y, center_x]
        b, g, r = int(b), int(g), int(r)
        return r, g, b
    
    def get_dominant_kmeans(self, img):
        pixels = img.reshape((-1, 3))
        pixels = numpy.float32(pixels)
        self._kmeans.fit(pixels)
        counts = Counter(self._kmeans.labels_)

        # Find the largest cluster and get its color
        dominant_color = self._kmeans.cluster_centers_[counts.most_common(1)[0][0]]
        return tuple(map(int, dominant_color))
    
    def get_color_name(self, R, G, B, color_path) -> str:
        csv = pandas.read_csv(f'./{color_path}', header = 0)
        minimum = 10000
        cname = None
        for i in range(len(csv)):
            d = abs(R - int(csv.loc[i, "r"])) + abs(G - int(csv.loc[i, "g"])) + abs(B - int(csv.loc[i, "b"]))
            if d < minimum:
                minimum = d
                cname, main_color = csv.loc[i, "color name"], csv.loc[i, "main color"]
        if cname is None:
            raise ValueError("No matching color founded in the dataset")
        return cname, main_color
        
    def get_color_main(self, img_path, color_path: str = '/data/colors_v3.csv',saturation_ratio: float = 1.2):
        img = cv2.imread(f"./{img_path}")
        if img is not None:
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w  = img.shape[:2]
            img = cv2.resize(img, (w//4, h//4), interpolation = cv2.INTER_AREA)
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv_img[:, :, 1] = numpy.clip(hsv_img[:, :, 1]* saturation_ratio, 0, 255)
            saturated_img= cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
            if self.strategy == 'centric':
                r, g, b = self.hicherachy_color(saturated_img, 2)
                _, main_color =  self.get_color_name(r, g, b, color_path)
                
            elif self.strategy == 'kmeans':
                b, g, r = self.get_dominant_kmeans(saturated_img)
                _, main_color = self.get_color_name(r,g,b, color_path)   
                
            else:
                raise ValueError("PLease choose one of the above strategy: centric, kmeans, hierachy")
        else: 
            raise ValueError("File not found")
            
        return main_color