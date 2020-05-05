import librosa

sr = librosa.get_samplerate('D:/Vasanth/DeepLearning/SubtitleSynchronizer/videos1/GardenofEvil.flac')

# Set the frame parameters to be equivalent to the librosa defaults
# in the file's native sampling rate
frame_length = (2048 * sr) // 22050
hop_length = (512 * sr) // 22050

# Stream the data, working on 128 frames at a time
stream = librosa.stream('D:/Vasanth/DeepLearning/SubtitleSynchronizer/videos1/GardenofEvil.flac',
                        block_length=128,
                        frame_length=frame_length,
                        hop_length=hop_length)

chromas = []
for y in stream:
   chroma_block = librosa.feature.chroma_stft(y=y, sr=sr,
                                              n_fft=frame_length,
                                              hop_length=hop_length,
                                              center=False)
   chromas.append(chromas)