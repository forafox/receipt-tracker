from pydantic import BaseModel, Field


class TrainingExample(BaseModel):
    text: str = Field(min_length=1)
    category: str = Field(min_length=1)
