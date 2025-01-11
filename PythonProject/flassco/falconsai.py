from transformers import pipeline

from helpers.long_text import GETLONGTEXT

summarizer = pipeline("summarization", model="Falconsai/text_summarization")

print(summarizer(GETLONGTEXT))
