import json
import re

def read_json(input):
    with open(input, "r", encoding = 'utf-8') as file:
        data = [json.loads(f) for f in file]
    return data

def save_json(input, output = 'output22.jsonl'):
    with open(output, "w", encoding="utf-8") as f:
        for entry in input:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

def remove_repeated_words(text):
    # Sử dụng regex để tìm các từ lặp lại nhiều lần
    cleaned_text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    return cleaned_text

def remove_incomplete_sentences(text):
    # Tách các câu dựa trên dấu chấm
    sentences = re.split(r'(?<=[.])\s+', text)
    # Giữ lại những câu kết thúc bằng dấu '.'
    filtered_sentences = [s for s in sentences if s.strip().endswith('.')]
    return " ".join(filtered_sentences)
