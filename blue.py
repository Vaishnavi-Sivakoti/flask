from nltk.translate.bleu_score import corpus_bleu
from cnnRnn import load_clean_descriptions

def calculate_bleu_scores(image_path, clip_sentence, cnn_sentence, comb_sentence):
    # Load reference descriptions for the test image
    reference_file = "descriptions.txt"  # Adjust the path accordingly
    reference_descriptions = load_clean_descriptions(reference_file, [image_path])
    
    # Convert reference descriptions to a list
    reference_list = [desc.split() for desc in reference_descriptions[image_path]]
    
    # Convert generated descriptions to lists
    generated_corpus_clip = clip_sentence.split()
    generated_corpus_cnn = cnn_sentence.split()
    generated_corpus_comb = comb_sentence.split()
    
    # Ensure that the lengths of the generated and reference lists are the same
    num_sentences = len(reference_list)
    generated_corpus_clip = [generated_corpus_clip] * num_sentences
    generated_corpus_cnn = [generated_corpus_cnn] * num_sentences
    generated_corpus_comb = [generated_corpus_comb] * num_sentences
    
    # Calculate BLEU score for the CLIP model
    bleu_score_clip = corpus_bleu(reference_list, generated_corpus_clip)
    
    # Calculate BLEU score for the CNN-RNN model
    bleu_score_cnn = corpus_bleu(reference_list, generated_corpus_cnn)

    # Calculate BLEU score for the Combined model
    bleu_score_comb = corpus_bleu(reference_list, generated_corpus_comb)
    
    return [bleu_score_clip, bleu_score_cnn, bleu_score_comb]
