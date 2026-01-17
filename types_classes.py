from typing import Literal
from pydantic import BaseModel
class request_type(BaseModel):
    follower_count: int
    date_time: str
    description: str
    hashtags: str
    

    