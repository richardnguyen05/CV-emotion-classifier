# emotion mapping dict
emotion_map = {
    '01' : 'neutral',
    '02' : 'calm',
    '03' : 'happy',
    '04' : 'sad',
    '05' : 'angry',
    '06' : 'fearful',
    '07' : 'disgust',
    '08' : 'surprised'
}

def parse_ravdess_filename(filename: str):
    """
    Parses a RAVDESS filename and returns a dictionary of its attributes.
    Example filename: '02-01-06-01-02-01-12.mp4'
    """
    parts = filename.split('-')
    return {
        'modality': parts[0],      # 02 = video, 03 = audio
        'vocal_channel': parts[1], # 01 = speech, 02 = song
        'emotion': parts[2],
        'emotion_label': emotion_map.get(parts[2], 'unknown'),
        'intensity': parts[3],
        'statement': parts[4],
        'repetition': parts[5],
        'actor': parts[6].split('.')[0]  # remove .wav or .mp4
    }