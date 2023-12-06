import os
import torchaudio
from speechbrain.pretrained import EncoderDecoderASR

# Deel 1: Laden en Transcriberen van Audio

# Instellingen voor torchaudio backend (verwijder indien niet nodig)
torchaudio.set_audio_backend("soundfile") 

# Pad naar de audiobestand en de transcriptie
audio_file_path = "C:\\Users\\msabi\\3_VS_GitProjects\\HMI_STT\\audio.mp3"
transcript_file_path = "C:\\Users\\msabi\\3_VS_GitProjects\\HMI_STT\\transcript.txt"

# Laad het audiobestand
waveform, sample_rate = torchaudio.load(audio_file_path)

# Initialiseer SpeechBrain model
asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-commonvoice-14-en", savedir="pretrained_models/speechbrain/asr-crdnn-commonvoice-14-en")

# Transcribeer de audio
transcription = asr_model.transcribe_file(audio_file_path)

# Opslaan van de transcriptie in hetzelfde folder als de audio
transcription_output_path = os.path.join(os.path.dirname(audio_file_path), "transcribed_text.txt")
with open(transcription_output_path, "w") as text_file:
    text_file.write(transcription)

# Vergelijk de transcribtie van SpeechBrain met de daadwerkelijke transcriptie
with open(transcript_file_path, "r") as file:
    ground_truth = file.read()

# Afdrukken van de resultaten (optioneel)
print("Transcription from SpeechBrain:\n", transcription)
print("\nGround Truth Transcription:\n", ground_truth)