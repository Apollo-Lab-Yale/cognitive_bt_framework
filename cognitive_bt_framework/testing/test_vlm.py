# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="lmsys/vicuna-7b-v1.1")