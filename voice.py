# needed installs :
# pip install speechbrain
# pip install soundfile
# pip install torchaudio
# pip install pydub
# pip install numpy
# ffmmpeg libs in os vars

import os
import re
import time
import torch
import torchaudio
from pydub import AudioSegment

import pandas as pd
import numpy as np
from collections import Counter

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

def align_texts(reference, hypothesis):

    """
    Check the hypothesis and reference and then align the texts so that the wer can be calculated

    Args:
    hypothesis (list): The predicted words.
    reference (list): The ground truth words.

    Returns:
    tuple: (align_ref, align_hyp)
    """

    # Normalize reference and hypothesis texts
    reference = preprocess(reference)
    hypothesis = preprocess(hypothesis)

    reference_words = reference.split()
    hypothesis_words = hypothesis.split()

    dp = [[0] * (len(hypothesis_words) + 1) for _ in range(len(reference_words) + 1)]
    # Based on  Wagner-Fischer algorithm
    for i in range(len(reference_words) + 1):
        for j in range(len(hypothesis_words) + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif reference_words[i - 1] == hypothesis_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i][j - 1], dp[i - 1][j])

    align_ref = []
    align_hyp = []
    i, j = len(reference_words), len(hypothesis_words)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference_words[i - 1] == hypothesis_words[j - 1]:
            align_ref.append(reference_words[i - 1])
            align_hyp.append(hypothesis_words[j - 1])
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j - 1] < dp[i - 1][j]):
            align_ref.append("")
            align_hyp.append(hypothesis_words[j - 1])
            j -= 1
        else:
            align_ref.append(reference_words[i - 1])
            align_hyp.append("")
            i -= 1

    align_ref.reverse()
    align_hyp.reverse()

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
        transcription = transcriptions_df[transcriptions_df['ID'] == int(task_id)]['Sentence'].values
        if transcription.size > 0:
            return transcription[0]

    return "No matching transcription found"

def calculate_recall_precision_fscore(hypothesis, reference):
    """
    Calculate Recall, Precision, and F-score for two lists of words.

    Args:
    hypothesis (list): The predicted words.
    reference (list): The ground truth words.

    Returns:
    tuple: (recall, precision, f_score)
    """

    # Count the occurrences of each word in both hypothesis and reference
    hyp_count = Counter(hypothesis)
    ref_count = Counter(reference)

    # True Positives: Words in hypothesis that are also in reference, counted distinctly
    true_positives = sum(min(hyp_count[word], ref_count[word]) for word in hyp_count)

    # False Positives: Words in hypothesis not in reference
    false_positives = sum(hyp_count[word] - min(hyp_count[word], ref_count[word]) for word in hyp_count)

    # False Negatives: Words in reference not in hypothesis
    false_negatives = sum(ref_count[word] - min(hyp_count[word], ref_count[word]) for word in ref_count)

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0

    # Calculate F-score
    f_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

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

    recall, precision, f_score = calculate_recall_precision_fscore(hypothesis=hypothesis, reference=reference)

    print("-----------------")
    print("Recall:", recall)
    print("Precision:", precision)
    print("F-score:", f_score)

def calculate_for_word(target_words, hypothesis):
    # Define target words for which you want to calculate precision and recall
    # target_words = []

    print("-----------------")
    print("Now performing calculations for the specified target words: ")

    # Calculate Recall and Precision for target words
    for word in target_words:
        true_positives = sum(1 for w in hypothesis if w == word)
        false_positives = len(hypothesis) - true_positives
        false_negatives = hypothesis.count(word) - true_positives

        recall = true_positives / (true_positives + false_negatives + 1e-9)  # Add a small epsilon to avoid division by zero
        precision = true_positives / (true_positives + false_positives + 1e-9)

        print("-----------------")
        print(f"Word: {word}")
        print("Recall:", recall)
        print("Precision:", precision)


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

reference_file_path = os.path.join(current_directory, reference_file)

df_ref = load_transcriptions_from_excel(reference_file_path)
audio_files = load_wav_files_from_folder(folder_path=os.path.join(current_directory, folder_name))

for audio in audio_files:
    reference = select_transcription_for_audio_file(audio_file=audio,transcriptions_df=df_ref)
    ai_magic(current_directory=current_directory, folder_name=folder_name, audio_file=audio, reference=reference)


