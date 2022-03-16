import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib

freq = 44100


def record(filename):
    duration = 4  # seconds
    print("Recording Audio")
    myrecording = sd.rec(duration * freq, samplerate=freq, channels=2, dtype='float64', device=2)
    sd.wait()
    print("Audio recording complete , Play Audio")
    sd.play(myrecording, freq)
    sd.wait()
    wav.write(filename, freq, myrecording)
    print("Play Audio Complete")
    return


def play_audio(filename):
    sample_rate, data = wav.read(filename)
    sd.play(data)
    sd.wait()
    return


def get_devices():
    print(sd.query_devices())
    return


def display_audio(filename, display):
    """
    Display either waveplot, fourier, spectogram, multi or mels
    """

    y, sr = librosa.load(filename)
    # trim
    data, _ = librosa.effects.trim(y)
    hop_length = 512
    n_fft = 2048
    n_mels = 128
    D = np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length))

    if display == 'waveplot':
        librosa.display.waveshow(data, sr=sr)
        plt.show()

    if display == 'fourier':
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear')
        plt.show()

    if display == 'spectogram':
        DB = librosa.amplitude_to_db(D, ref=np.max)
        librosa.display.specshow(DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.show()

    if display == 'multi':
        mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        plt.figure(figsize=(15, 4))

        # Plotting Hz to mels
        plt.subplot(1, 4, 1)
        librosa.display.specshow(mel, sr=sr, hop_length=hop_length, x_axis='linear')
        plt.ylabel('Mel filter')
        plt.colorbar()
        plt.title('Hz to mels.')

        # Dropping to 10 mels to see better
        plt.subplot(1, 4, 2)
        mel_10 = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=10)
        librosa.display.specshow(mel_10, sr=sr, hop_length=hop_length, x_axis='linear')
        plt.ylabel('Mel filter')
        plt.colorbar()
        plt.title('only 10 mels.')

        # Hz scale partitioned into bins, transformed for Mel Scale using overlapping triangular filters
        plt.subplot(1, 4, 3)
        idxs_to_plot = [0, 9, 49, 99, 127]
        for i in idxs_to_plot:
            plt.plot(mel[i])
        plt.legend(labels=[f'{i + 1}' for i in idxs_to_plot])
        plt.title('triangular filters')
        plt.tight_layout()

        plt.subplot(1, 4, 4)
        plt.plot(D[:, 1])
        plt.plot(mel.dot(D[:, 1]))
        plt.legend(labels=['Hz', 'mel'])
        plt.title('before and after converting to mel.')
        plt.isinteractive()
        plt.show()

    if display == 'mels':
        S = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.show()


def compare_mels(file_list):
    num_files = len(file_list)

    for x, filename in enumerate(file_list):
        y, sr = librosa.load(filename)
        data, _ = librosa.effects.trim(y)
        hop_length = 512
        n_fft = 2048
        n_mels = 128
        fig = plt.subplot(1, num_files, x+1)
        S = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        fig.plot()

    plt.show()


if __name__ == '__main__':
    # display_audio(filename='name.wav', display='multi')
    compare_mels(file_list= ['name.wav', 'name_1.wav', 'name_2.wav', 'name_3.wav'])

