import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    model_name: str = 'google/gemini-3-flash-pre-thinking'

    tavily_api_key: str = os.getenv('TAVILY_API_KEY')
    vsegpt_api_key: str = os.getenv('VSEGPT_API_KEY')


config = Config()
