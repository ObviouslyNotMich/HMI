# needed installs :
# pip install speechbrain
# pip install soundfile
# pip install torchaudio
# pip install pydub
# pip install numpy
# ffmmpeg libs in os vars
# pip install openpyxl

import os
import re
import time
import torch
import torchaudio
from pydub import AudioSegment

import pandas as pd
import numpy as np
from collections import Counter
from difflib import SequenceMatcher

from speechbrain.pretrained import EncoderDecoderASR

os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'
current_directory = os.path.dirname(__file__)
folder_name = 'audio'

# Load ground truth from a .excel file
reference_file = "speechrecognition dataset details.xlsx"  

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


# Wer calculation from the powerpoint slides
def calculate_wer(ref_words, hyp_words):

	# Counting the number of substitutions, deletions, and insertions
	substitutions = sum(1 for ref, hyp in zip(ref_words, hyp_words) if ref != hyp)
	deletions = len(ref_words) - len(hyp_words)
	insertions = len(hyp_words) - len(ref_words)
 
	# Total number of words in the reference text
	total_words = len(ref_words)
 
	# Calculating the Word Error Rate (WER)
	wer = (substitutions + deletions + insertions) / total_words
	return wer

# Wer calculation that way more accurate acording to python book and numpy website
def calculate_wer2(ref_words, hyp_words):
    # Initialize a matrix with size |ref_words|+1 x |hyp_words|+1
    # The extra row and column are for the case when one of the strings is empty
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    # The number of operations for an empty hypothesis to become the reference
    # is just the number of words in the reference (i.e., deleting all words)
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    # The number of operations for an empty reference to become the hypothesis
    # is just the number of words in the hypothesis (i.e., inserting all words)
    for j in range(len(hyp_words) + 1):
        d[0, j] = j
    # Iterate over the words in the reference and hypothesis
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            # If the current words are the same, no operation is needed
            # So we just take the previous minimum number of operations
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                # If the words are different, we consider three operations:
                # substitution, insertion, and deletion
                # And we take the minimum of these three possibilities
                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)
    # The minimum number of operations to transform the hypothesis into the reference is in the bottom-right cell of the matrix We divide this by the number of words in the reference to get the WER
    wer = d[len(ref_words), len(hyp_words)] / len(ref_words)
    return wer

def preprocess(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation and extra whitespaces
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def similar(a, b):
    # Define a similarity threshold
    threshold = 0.5
    return SequenceMatcher(None, a, b).ratio() > threshold

def find_best_alignment(current_word, other_words, look_ahead_limit=2):
    for offset in range(1, min(look_ahead_limit + 1, len(other_words))):
        if similar(current_word, other_words[offset]):
            return offset
    return 0

def align_texts(reference, hypothesis):
    # Normalize reference and hypothesis texts
    reference = preprocess(reference)
    hypothesis = preprocess(hypothesis)

    reference_words = reference.split()
    hypothesis_words = hypothesis.split()

    align_ref, align_hyp = [], []
    i, j = 0, 0

    while i < len(reference_words) and j < len(hypothesis_words):
        if reference_words[i] == hypothesis_words[j]:
            align_ref.append(reference_words[i])
            align_hyp.append(hypothesis_words[j])
            i += 1
            j += 1
        elif similar(reference_words[i], hypothesis_words[j]):
            align_ref.append(reference_words[i])
            align_hyp.append(hypothesis_words[j])
            i += 1
            j += 1
        else:
            ref_look_ahead = find_best_alignment(reference_words[i], hypothesis_words[j:], 3)
            hyp_look_ahead = find_best_alignment(hypothesis_words[j], reference_words[i:], 3)

            if ref_look_ahead > 0:
                align_ref.append("")
                align_hyp.append(hypothesis_words[j])
                j += 1
            elif hyp_look_ahead > 0:
                align_ref.append(reference_words[i])
                align_hyp.append("")
                i += 1
            else:
                align_ref.append(reference_words[i])
                align_hyp.append("")
                i += 1

    # Handle remaining words in either list
    while i < len(reference_words):
        align_ref.append(reference_words[i])
        align_hyp.append("")
        i += 1

    while j < len(hypothesis_words):
        align_ref.append("")
        align_hyp.append(hypothesis_words[j])
        j += 1

    # Printing aligned words side by side for visual confirmation
    print("align_ref - align_hyp")
    for ref, hyp in zip(align_ref, align_hyp):
        print(f"{ref.ljust(15)} - {hyp}")

    return align_ref, align_hyp

def load_wav_files_from_folder(folder_path):
    """
    Function to load all .wav files from a specified folder.

    Args:
    folder_path (str): Path to the folder containing the .wav files.

    Returns:
    list: A list of .wav file names found in the folder.
    """
    wav_files = []

    # Iterate over files in the specified folder
    for file in os.listdir(folder_path):
        # Check if the file is a .wav file
        if file.endswith(".wav"):
            wav_files.append(file)

    return wav_files

def load_transcriptions_from_excel(excel_path):
    """
    Function to load transcription ground truths from an Excel file.

    Args:
    excel_path (str): Path to the Excel file.

    Returns:
    DataFrame: A pandas DataFrame containing the transcriptions with 'ID' and 'Sentence' columns.
    """
    # Read the Excel file into a DataFrame
    try:
        df = pd.read_excel(excel_path, usecols=['ID', 'Sentence'])

        # Cleaning the DataFrame to remove unnecessary columns
        df = df[['ID', 'Sentence']].dropna()

        return df
    except Exception as e:
        print(e)
        return f"An error occurred: {e}"

def select_transcription_for_audio_file(audio_file, transcriptions_df):
    """
    Function to select the correct transcription for a given audio file based on the task number in the file name.

    Args:
    audio_file (str): The audio file name.
    transcriptions_df (DataFrame): DataFrame containing transcription data with 'ID' and 'Sentence' columns.

    Returns:
    str: The corresponding transcription sentence for the given audio file.
    """
    # Extract the task ID from the audio file name
    match = re.search(r"task(\d+)_", audio_file)
    if match:
        task_id = int(match.group(1))  # Convert the extracted ID to an integer

        # Find the corresponding transcription
        transcription = transcriptions_df[transcriptions_df['ID'] == task_id]['Sentence'].values
        if transcription.size > 0:
            return transcription[0]

    return "No matching transcription found"

def calculate_recall_precision_fscore(hypothesis, reference):
    # Count occurrences in both lists
    hyp_count = Counter(hypothesis)
    ref_count = Counter(reference)

    # Calculate true positives (words in both hypothesis and reference)
    true_positives = 0

    for index, element in enumerate(hypothesis):
        if hypothesis[index] == reference[index]:
            true_positives += 1

    # Exclude empty strings from the hypothesis count
    if "" in hyp_count:
        del hyp_count[""]

    # Exclude empty strings from the reference count
    if "" in ref_count:
        del ref_count[""]

    # Calculate total number of predicted and actual words
    total_predicted = sum(hyp_count.values())
    total_actual = sum(ref_count.values())

    # Calculate precision and recall
    precision = true_positives / total_predicted if total_predicted > 0 else 0
    recall = true_positives / total_actual if total_actual > 0 else 0

    # Calculate F-score
    if recall + precision > 0:
        f_score = 2 * (precision * recall) / (precision + recall)
    else:
        f_score = 0

    return recall, precision, f_score

def ai_magic(current_directory,folder_name ,audio_file, reference):

    path = os.path.join(current_directory, folder_name, audio_file)

    waveform, sample_rate = torchaudio.load(path)

    # Run speech recognition on GPU
    run_opts={"device":"cuda"}

    # Initialize the speech recognition model
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech", savedir="pretrained_models/asr-transformer-transformerlm-librispeech", run_opts=run_opts)
    # asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-commonvoice-14-en", savedir="pretrained_models/speechbrain/asr-crdnn-commonvoice-14-en", run_opts=run_opts)

    print("-----------------")
    print("Loaded Model OK - Performing Transcription on:   " + audio_file)
    print("-----------------")

    # Start timing the transcription
    start_time = time.time()

    # Perform transcription
    transcription = asr_model.transcribe_file(path)

    # End timing the transcription
    end_time = time.time()

    # Calculate transcription time and audio duration
    transcription_time = end_time - start_time

    audio_duration = len(waveform[0]) / sample_rate  # Calculate audio duration in seconds

    # Calculate Real-Time Factor (RTF)
    rtf = transcription_time / audio_duration

    print("-----------------")
    print("Transcription Time:", transcription_time, "seconds")
    print("Audio Duration:", audio_duration, "seconds")
    print("Real Time Factor (RTF):", rtf)
    print(" ")
    print("Reference: " + reference)
    print(" ")
    print("Hypothesis: " + transcription)

    reference, hypothesis  = align_texts(reference=reference, hypothesis=transcription)
    

    wer = float(calculate_wer2(ref_words=reference, hyp_words=hypothesis))
    wcr = 1 - wer

    print("-----------------")
    print("Word Error Rate (WER):", calculate_wer(ref_words=reference, hyp_words=hypothesis))
    print("Accurate Word Error Rate (AWER):", wer)
    print("Word Correct Rate (WCR):", wcr)

    # Calculate Levenshtein Distance
    lev_distance = levenshtein_distance(" ".join(reference), " ".join(hypothesis))
    print("-----------------")
    print("Levenshtein Distance:", lev_distance)

    recall, precision, f_score = calculate_recall_precision_fscore(hypothesis=hypothesis, reference=reference)

    print("-----------------")
    print("Recall:", recall)
    print("Precision:", precision)
    print("F-score:", f_score)

def calculate_for_word(target_words, hypothesis, reference):
    print("-----------------")
    print("Now performing calculations for the specified target words: ")

    hyp_count = Counter(hypothesis)
    ref_count = Counter(reference)

    for word in target_words:
        tp = 0
        for index, element in enumerate(hypothesis):
            if element == word and hypothesis[index] == reference[index]:
              tp += 1


        word_hyp_count = hyp_count[word]
        word_ref_count = ref_count[word]

        # Recall for the word
        recall = tp / word_ref_count if word_ref_count > 0 else 0

        # Precision for the word
        precision = tp / word_hyp_count if word_hyp_count > 0 else 0

        # F-score for the word
        if recall + precision > 0:
            f_score = 2 * (precision * recall) / (precision + recall)
        else:
            f_score = 0

        print("-----------------")
        print(f"Word: {word}")
        print("Recall:", recall)
        print("Precision:", precision)
        print("F-score:", f_score)


def mp3_to_wav():
    # Check if the file is mp3
    if file_path.endswith('.mp3'):
        # Convert mp3 to wav
        audio = AudioSegment.from_mp3(file_path)
        # Change the file path to wav
        file_path = file_path.replace('.mp3', '.wav')
        # Export as wav
        audio.export(file_path, format='wav')

    # Load file in wav since we are windows
    return torchaudio.load(file_path)


def fake_it(reference, hypothesis):
    print("-----------------")
    print("Reference: " + reference)
    print(" ")
    print("Hypothesis: " + hypothesis)

    reference, hypothesis  = align_texts(reference=reference, hypothesis=hypothesis)
    

    wer = calculate_wer2(ref_words=reference, hyp_words=hypothesis)
    wcr = 1 - wer

    print("-----------------")
    print("Word Error Rate (WER):", calculate_wer(ref_words=reference, hyp_words=hypothesis))
    print("Accurate Word Error Rate (AWER):", wer)
    print("Word Correct Rate (WCR):", wcr)

    # Calculate Levenshtein Distance
    lev_distance = levenshtein_distance(" ".join(reference), " ".join(hypothesis))
    print("-----------------")
    print("Levenshtein Distance:", lev_distance)


    calculate_for_word(hypothesis=hypothesis, reference=reference, target_words=["our", "the"])
    
    recall, precision, f_score = calculate_recall_precision_fscore(hypothesis=hypothesis, reference=reference)

    print("-----------------")
    print("Macro Recall:", recall)
    print("Macro Precision:", precision)
    print("Macro F-score:", f_score)


fake_ref = "Additionally, our innovative pipeline includes more than twenty active development programs for blood cancers and solid tumors, which we expect will strengthen our growing position in oncology."
fake_hypo = "Addition ally, or innovative pipeline includes more than twenty active develop mint programs for blood cancers solid tumors, which we will strengthen our glowing positioning oncology"
pp_ref = "The cat sat on the mat at the door"
pp_hypo = "She rat the sat the mat at door"


fake_it(reference=pp_ref, hypothesis=pp_hypo)

# reference_file_path = os.path.join(current_directory, reference_file)

# df_ref = load_transcriptions_from_excel(reference_file_path)
# audio_files = load_wav_files_from_folder(folder_path=os.path.join(current_directory, folder_name))

# for audio in audio_files:
#     reference = select_transcription_for_audio_file(audio_file=audio,transcriptions_df=df_ref)
#     ai_magic(current_directory=current_directory, folder_name=folder_name, audio_file=audio, reference=reference)
#     torch.cuda.empty_cache()