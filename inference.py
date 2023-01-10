from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    pipeline
    )


class Translator:

    def __init__(self):
        self.checkpoint = "facebook/nllb-200-distilled-600M" #nllb-200-distilled-1.3B 
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.source = "sun_Latn"
        self.target = "ind_Latn"
        self.mode = "translation"
        self.length = 100 #400 max
        self.translator = pipeline(
            task = self.mode, 
            model = self.model, 
            tokenizer = self.tokenizer, 
            src_lang = self.source, 
            tgt_lang = self.target, 
            max_length = self.length
            )

    def translate(self, text):
        return self.translator(text)[0]['translation_text']

if __name__ == "__main__":
    translate_model = Translator()
    sample_text = "abdi ka sakola kamari"
    result_text = translate_model.translate(text = sample_text)
    print(f"Sundanese : {sample_text}")
    print(f"Indonesia : {result_text}")
