import spacy
from transformers import BartForConditionalGeneration, BartTokenizer
def remove_words(sentence, words_to_remove):
    print(words_to_remove)
    # Split the sentence into words
    words = sentence.split()

    # Filter out the words to remove
    filtered_words = [word for word in words if word not in words_to_remove]

    # Join the filtered words back into a sentence
    new_sentence = " ".join(filtered_words)

    return new_sentence

def identify_shared_tokens(sentence1, sentence2):
    nlp = spacy.load("en_core_web_sm")
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)
    tokens1 = set(token.text for token in doc1 if token.pos_ in ["NOUN", "VERB"])
    tokens2 = set(token.text for token in doc2 if token.pos_ in ["NOUN", "VERB"])
    shared_tokens = tokens1.intersection(tokens2)
    
    # Remove common verbs from sentence2
    common_verbs = set(token.text for token in doc1 if token.pos_ == "VERB") & set(token.text for token in doc2 if token.pos_ == "VERB")
    tokens2 -= common_verbs
    
    return shared_tokens.union(tokens1)

def combine_sentences(sentence1, sentence2):
    shared_tokens = identify_shared_tokens(sentence1, sentence2)
    print(shared_tokens)
    if shared_tokens:
        if len(shared_tokens) > 1:
            combined_sentence = f"{sentence1} and {sentence2}"
        else:
            combined_sentence = f"{sentence1} with {remove_words(sentence2,shared_tokens)}"
            token = shared_tokens.pop()
            
    else:
        combined_sentence = sentence1
    return combined_sentence

def generate_combined_sentence_with_bart(sentence1, sentence2):
    combined_sentence = combine_sentences(sentence1, sentence2)
    if not combined_sentence:
        prompt = f"Combine the sentences: '{sentence1}' and '{sentence2}'"
        model_name = "facebook/bart-large-cnn"
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1, early_stopping=True)
        combined_sentence = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return combined_sentence
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk
stop_words = set(nltk.corpus.stopwords.words('english'))

def tokenize(sentence):
    return sentence.lower().split()

# Generate new sentence
def generate_new_sentence(sentence1, sentence2):
    # Load English stop words

    # Tokenization (replace with your preferred tokenizer if needed)
    tokens1 = [word for word in sentence1.lower().split() if word not in stop_words]
    tokens2 = [word for word in sentence2.lower().split() if word not in stop_words]

    # Create vocabulary
    vocab = set(tokens1 + tokens2)

    # Vectorization using Bag of Words
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False)
    sentence_vectors = vectorizer.fit_transform([sentence1, sentence2]).toarray()

    # Calculate cosine similarity
    cosine_similarity = np.dot(sentence_vectors[0], sentence_vectors[1]) / (
        np.linalg.norm(sentence_vectors[0]) * np.linalg.norm(sentence_vectors[1])
    )

    # Similarity threshold (adjust as needed)
    similarity_threshold = 0.4

    if cosine_similarity >= similarity_threshold:
        print(f"Sentences are similar (cosine similarity: {cosine_similarity:.2f}).")
        return sentence1  # Return the more likely similar sentence
    else:
        print(f"Sentences are not similar enough (cosine similarity: {cosine_similarity:.2f}).")
        # Consider using a more advanced technique like BART for combining sentences
        # (implementation omitted for brevity and potential resource limitations)
        return generate_combined_sentence_with_bart(sentence1,sentence2)  # Placeholder
