from pydantic import BaseModel

class InferenceInput(BaseModel):
    user_input: str

class InferenceOutput(BaseModel):
    inference_output: str