import sys
import os, time
import datetime
from datetime import timezone, timedelta
from typing import Optional, Union, Dict, Any
from pathlib import Path

from enum import Enum
from loguru import logger as log
from pydantic import BaseModel, validator, root_validator
os.environ['TZ'] = 'Asia/Ho_Chi_Minh'
# time.tzset()
# os.environ['TZ'] = zone

class Logging:
    """
    A utility class for handling logging operations using Loguru.
    Provides methods for configuring log outputs, setting log levels,
    and managing log rotation with UTC+7 timezone support.
    """
    def __init__(self,
                 log_file: Optional[Union[str, Path]] = None,
                 log_level: str = "INFO",
                 rotation: str = "1 day",
                 retention: str = "1 week",
                 compression: str = "zip"):
        """
        Initialize logging configuration.
        
        Args:
            log_file: Path to log file. If None, logs only to stderr
            log_level: Minimum log level to record
            rotation: When to rotate the log file
            retention: How long to keep log files
            compression: Compression format for rotated logs
        """
        # Store logger instance
        self.log = log
        
        # Remove default logger
        self.log.remove()
        
        self.format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss} UTC+7</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        
        # Configure stderr logging with time converter
        self.log.add(
            sys.stderr,
            format=self.format,
            level=log_level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )
        
        # Configure file logging if path is provided
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            self.log.add(
                str(log_file),
                format=self.format,
                level=log_level,
                rotation=rotation,
                retention=retention,
                compression=compression,
                enqueue=True,
                backtrace=True,
                diagnose=True,
            )
    
    def set_level(self, level: str) -> None:
        """Change the logging level."""
        self.log.level(level)
    
    def add_context(self, **kwargs: Any) -> None:
        """Add contextual information to log messages."""
        for key, value in kwargs.items():
            self.log.bind(**{key: value})    
    def debug(self, message: Union[str, Any], *args: Any, **kwargs: Any) -> None:
        """Log debug message"""
        if isinstance(message, str):
            self.log.debug(message.format(*args), **kwargs)
        else:
            self.log.debug(str(message), **kwargs)
        
    def info(self, message:  Union[str, Any], *args: Any, **kwargs: Any) -> None:
        """Log info message"""
        if isinstance(message, str):
            if args:  # Chỉ format nếu có tham số để thay thế
                self.log.info(message.format(*args), **kwargs)
            else:
                self.log.info(message, **kwargs)
        else:
            self.log.info(str(message), **kwargs)
        
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message"""
        self.log.warning(message, **kwargs)
        
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.log.error(message, **kwargs)
        
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.log.critical(message, **kwargs)
        
        
class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogConfig(BaseModel):
    base_log_dir: Path = Path("logs")
    log_file: Optional[Union[str, Path]] = None
    log_level: LogLevel = LogLevel.INFO
    rotation: str = "1 day"
    retention: str = "1 week"
    compression: str = "zip"
    
    @root_validator(pre=True)
    def adjust_log_file(cls, values):
        base_log_dir = values.get("base_log_dir", Path("logs"))
        log_file = values.get("log_file")

        # Ensure base_log_dir exists
        if not isinstance(base_log_dir, Path):
            base_log_dir = Path(base_log_dir)
        base_log_dir.mkdir(parents=True, exist_ok=True)

        # Adjust log_file path
        if log_file is None:
            log_file = base_log_dir / "default.log"  # Default log file name
        else:
            log_file = base_log_dir / Path(log_file).name  # Ensure it's inside base_log_dir

        values["log_file"] = log_file
        return values

    @validator('rotation')
    def validate_rotation(cls, v):
        valid_units = ['day', 'week', 'month', 'year']
        if not any(unit in v.lower() for unit in valid_units):
            raise ValueError(f"rotation must contain one of {valid_units}")
        return v
    
    @validator('retention')
    def validate_retention(cls, v):
        valid_units = ['day', 'week', 'month', 'year']
        if not any(unit in v.lower() for unit in valid_units):
            raise ValueError(f"retention must contain one of {valid_units}")
        return v