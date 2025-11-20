import numpy as np
import matplotlib.pyplot as plt
import librosa

def plot_summed_harmonics(audio_path, num_harmonics):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Compute FFT
    fft = np.fft.fft(y)
    magnitude = np.abs(fft)[:len(fft)//2]  # Only positive frequencies
    phase = np.angle(fft)[:len(fft)//2]  # Get phase information
    frequency = np.linspace(0, sr/2, len(magnitude))  # Frequency axis

    # Find peaks (harmonics)
    peak_indices = np.argsort(magnitude)[-num_harmonics:]  # Get indices of top harmonics
    peak_freqs = frequency[peak_indices]  # Get corresponding frequencies
    peak_magnitudes = magnitude[peak_indices]  # Get corresponding magnitudes

    # Generate time vector for plotting
    time = np.linspace(0, 20, sr * 2) 

    # Start with a zero wave to accumulate harmonics
    summed_wave = np.zeros_like(time)

    plt.figure(figsize=(12, 6))

    for i, f in enumerate(peak_freqs):
        # Extract the real and imaginary parts from the FFT (cosine and sine components)
        A_n = peak_magnitudes[i] * np.cos(phase[i])  # Real part (cosine coefficient)
        B_n = peak_magnitudes[i] * np.sin(phase[i])  # Imaginary part (sine coefficient)
        
        # Reconstruct the harmonic using A_n and B_n (cosine and sine)
        harmonic_wave = A_n * np.cos(2 * np.pi * f * time) + B_n * np.sin(2 * np.pi * f * time)
        summed_wave += harmonic_wave  # Sum the harmonic to the total wave
    
    # Plot the final summed wave
    plt.plot(time[:1000], summed_wave[:1000], color="black", linewidth=2)

    #plt.xlabel("Time (seconds)")
    #plt.ylabel("Amplitude")
    #plt.title("Summed Harmonics Waveform")
    plt.show()

# Call function
plot_summed_harmonics("05:04:2025.mp3", num_harmonics=3)