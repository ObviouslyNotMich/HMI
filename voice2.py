# needed installs :
# pip install speechbrain
# pip install soundfile
# pip install torchaudio
# pip install pydub
# pip install numpy
# ffmmpeg libs in os vars

import os
import time

os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'
current_directory = os.path.dirname(__file__)

import torch
import torchaudio
from pydub import AudioSegment

import numpy as np

from speechbrain.pretrained import EncoderDecoderASR

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


# Load ground truth from a .txt file
reference_file = "trans.txt"  # Replace with your ground truth .txt file path

reference_file = os.path.join(current_directory, reference_file)

with open(reference_file, "r") as file:
    reference = file.read()

# Define the folder name where the audio file is located
folder_name = 'audio'

# Define the file name of the audio file
file_name = 'audio.mp3'

# Construct the file path
file_path = os.path.join(current_directory, folder_name, file_name)

# Check if the file is mp3
if file_path.endswith('.mp3'):
    # Convert mp3 to wav
    audio = AudioSegment.from_mp3(file_path)
    # Change the file path to wav
    file_path = file_path.replace('.mp3', '.wav')
    # Export as wav
    audio.export(file_path, format='wav')

# Load file in wav since we are windows
waveform, sample_rate = torchaudio.load(file_path)

# Run speech recognition on GPU
run_opts={"device":"cuda"}

# Initialize the speech recognition model
asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech", savedir="pretrained_models/asr-transformer-transformerlm-librispeech", run_opts=run_opts)
# asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-commonvoice-14-en", savedir="pretrained_models/speechbrain/asr-crdnn-commonvoice-14-en", run_opts=run_opts)

print("Loaded Model OK - Performing Transcription")

# Start timing the transcription
start_time = time.time()

# Perform transcription
transcription = asr_model.transcribe_file(file_path)

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

hypothesis = transcription.lower().split()
reference = reference.lower().split()

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

# Followed this https://www.askpython.com/python/examples/precision-and-recall-in-python
# Calculate Recall, Precision, and F-score
true_positives = sum(1 for word in hypothesis if word in reference)
false_positives = len(hypothesis) - true_positives
false_negatives = len(reference) - true_positives

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)

# Divide by zero fix
if precision + recall != 0:
    f_score = 2 * (precision * recall) / (precision + recall)
else:
    f_score = 0

print("-----------------")
print("Recall:", recall)
print("Precision:", precision)
print("F-score:", f_score)

# Define target words for which you want to calculate precision and recall
target_words = ["that", "dream"]

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
