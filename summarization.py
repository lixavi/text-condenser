from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def generate_summary(text):
    # Load pre-trained model and tokenizer
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Tokenize input text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # Decode summary tokens back to text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary
