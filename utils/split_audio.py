from pydub import AudioSegment
from etc import GENERATED_DATA_PATH


def split_audio(audio_entry, transcriptions_desc, audio_output_path):

    for i, transcript in enumerate(transcriptions_desc):
        start_time = transcript.start_time * 1000
        end_time = transcript.end_time * 1000
        audio = AudioSegment.from_wav(audio_entry)
        audio = audio[start_time:end_time]
        file_name = audio_output_path+"audio"+str(i)+".wav"
        audio.export(file_name, format="wav")




