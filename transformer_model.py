from transformers import T5ForConditionalGeneration, T5Tokenizer

class TransformerSummarizer:
    def __init__(self, model_name="t5-small"):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def summarize(self, text, max_length=150, min_length=40, length_penalty=2.0, num_beams=4):
        input_ids = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=False)
        
        # Generate summary
        summary_ids = self.model.generate(input_ids, max_length=max_length, min_length=min_length, 
                                           length_penalty=length_penalty, num_beams=num_beams, 
                                           early_stopping=True)
        
        # Decode summary tokens back to text
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary
