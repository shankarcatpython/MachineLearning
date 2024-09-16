import os
import sys
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification, ElectraForCausalLM
import torch

# Disable Hugging Face symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Set UTF-8 encoding for the console to avoid UnicodeEncodeError
sys.stdout.reconfigure(encoding='utf-8')

class NLPModelPipeline:
    def __init__(self):
        pass

    @staticmethod
    def load_pipeline(task, model_name):
        try:
            return pipeline(task, model=model_name)
        except Exception as e:
            print(f"Error loading model {model_name} for task {task}: {e}")
            return None

    @staticmethod
    def load_model_and_tokenizer(model_name):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            return model, tokenizer
        except Exception as e:
            print(f"Error loading model and tokenizer {model_name}: {e}")
            return None, None

    # 1. DistilBERT for Sentiment Analysis
    def distilbert_sentiment_analysis(self):
        classifier = self.load_pipeline('sentiment-analysis', 'distilbert-base-uncased-finetuned-sst-2-english')
        if classifier:
            result = classifier("I love using small language models!")
            print("DistilBERT Sentiment Analysis:", result)

    # 2. TinyBERT for Named Entity Recognition (NER)
    def tinybert_ner(self):
        model, tokenizer = self.load_model_and_tokenizer("dbmdz/bert-large-cased-finetuned-conll03-english")
        if model and tokenizer:
            inputs = tokenizer("OpenAI is located in San Francisco.", return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
            print("TinyBERT Named Entity Recognition:", predictions)

    # 3. ALBERT for Semantic Similarity Detection
    def albert_semantic_similarity(self):
        classifier = self.load_pipeline('text-classification', 'textattack/albert-base-v2-snli')
        if classifier:
            result = classifier("A soccer game with multiple males playing. [SEP] Some men are playing a sport.")
            print("ALBERT Semantic Similarity Detection:", result)

    # 4. ELECTRA-Small for Text Generation
    def electra_text_generation(self):
        model = ElectraForCausalLM.from_pretrained('google/electra-small-discriminator', is_decoder=True)
        generator = pipeline('text-generation', model=model)
        if generator:
            generated_text = generator("In the future of artificial intelligence, ", max_length=50, truncation=True)
            print("ELECTRA-Small Text Generation:", generated_text)

    # 5. MobileBERT for On-Device Text Classification
    def mobilebert_text_classification(self):
        classifier = self.load_pipeline('text-classification', 'google/mobilebert-uncased')
        if classifier:
            result = classifier("This is a mobile-friendly NLP model.")
            print("MobileBERT Text Classification:", result)

    # 6. MiniLM for Document Ranking and Search
    def minilm_document_ranking(self):
        tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
        model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
        ranker = pipeline('text-classification', model=model, tokenizer=tokenizer)
        if ranker:
            results = ranker("What is artificial intelligence? [SEP] AI is the simulation of human intelligence.")
            print("MiniLM Document Ranking and Search:", results)

    # 7. TinyGPT for Text Completion
    def tinygpt_text_completion(self):
        generator = self.load_pipeline('text-generation', 'huggingface/tiny-gpt2')
        if generator:
            generated_text = generator("Machine learning is transforming industries by", max_length=50, truncation=True)
            print("TinyGPT Text Completion:", generated_text)

    # 8. DistilRoBERTa for Fake News Detection
    def distilroberta_fake_news_detection(self):
        classifier = self.load_pipeline('text-classification', 'mrm8488/distilroberta-finetuned-fakenews')
        if classifier:
            result = classifier("The moon landing in 1969 was faked by NASA.")
            print("DistilRoBERTa Fake News Detection:", result)

    # 9. LiteGPT for Dialogue Generation
    def litegpt_dialogue_generation(self):
        generator = self.load_pipeline('text-generation', 'distilgpt2')
        if generator:
            response = generator("User: Hello! How are you?\nBot:", max_length=50, truncation=True)
            print("LiteGPT Dialogue Generation:", response)

    # 10. BERT-Tiny for Text Classification
    def berttiny_text_classification(self):
        classifier = self.load_pipeline('text-classification', 'prajjwal1/bert-tiny')
        if classifier:
            result = classifier("I feel great today!")
            print("BERT-Tiny Text Classification:", result)

    # 11. CamemBERT-Tiny for French Sentiment Analysis
    def camembert_french_sentiment_analysis(self):
        classifier = self.load_pipeline('sentiment-analysis', 'cmarkea/distilcamembert-base')
        if classifier:
            result = classifier("C'est une excellente journée!")
            print("CamemBERT-Tiny French Sentiment Analysis:", result)

    # 12. Funnel Transformer for Text Summarization
    def funnel_transformer_text_summarization(self):
        summarizer = self.load_pipeline('summarization', 'funnel-transformer/small')
        if summarizer:
            summary = summarizer("Long document text goes here...", max_length=50, truncation=True)
            print("Funnel Transformer Text Summarization:", summary)

    # 13. Reformer for Language Modeling
    def reformer_language_modeling(self):
        generator = self.load_pipeline('text-generation', 'google/reformer-enwik8')
        if generator:
            output = generator("The importance of data in AI is", max_length=50, truncation=True)
            print("Reformer Language Modeling:", output)

    # 14. SqueezeBERT for Intent Detection
    def squeezebert_intent_detection(self):
        classifier = self.load_pipeline('text-classification', 'squeezebert/squeezebert-uncased')
        if classifier:
            result = classifier("Book a flight to New York.")
            print("SqueezeBERT Intent Detection:", result)

if __name__ == "__main__":
    nlp_pipeline = NLPModelPipeline()
    nlp_pipeline.distilbert_sentiment_analysis()
    nlp_pipeline.tinybert_ner()
    nlp_pipeline.albert_semantic_similarity()
    nlp_pipeline.electra_text_generation()
    nlp_pipeline.mobilebert_text_classification()
    nlp_pipeline.minilm_document_ranking()
    nlp_pipeline.tinygpt_text_completion()
    nlp_pipeline.distilroberta_fake_news_detection()
    nlp_pipeline.litegpt_dialogue_generation()
    nlp_pipeline.berttiny_text_classification()
    nlp_pipeline.camembert_french_sentiment_analysis()
    nlp_pipeline.funnel_transformer_text_summarization()
    nlp_pipeline.reformer_language_modeling()
    nlp_pipeline.squeezebert_intent_detection()
