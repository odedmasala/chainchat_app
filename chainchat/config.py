from pydantic_settings import BaseSettings
from pydantic import model_validator
from typing import Optional
import os


class Settings(BaseSettings):
    
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    
    app_name: str = "ChainChat"
    max_tokens: int = 2000
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_file_size: int = 100 * 1024 * 1024
    
    secret_key: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"
    
    @model_validator(mode='after')
    def validate_openai_key(self):
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required. Please set it in your .env file.")
        return self


settings = Settings() 