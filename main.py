# -*- coding: utf-8 -*-
"""
Estimate Relaxation from Band Powers

pip install simpleaudio pydub pylsl
pip install sounddevice
pip install MIDIUtil
pip install mido python-rtmidi
pip install --upgrade diffusers accelerate transformers
pip install matplotlib
pip install scikit-learn

Run:
muselsl stream
python3 main.py

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylsl import StreamInlet, resolve_byprop
import utils

from pydub import AudioSegment
from pydub.generators import Sine
from pydub.playback import play
import sounddevice as sd
import mido
from mido import Message, get_output_names
import time


class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3


BUFFER_LENGTH = 5
EPOCH_LENGTH = 1
OVERLAP_LENGTH = 0.8
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH
INDEX_CHANNEL = [0]


class ArtGenerator:
    def __init__(self, data, img):
        self.data = data
        self.img = img

    def cellular_automaton(self):
        # Create a random initial state
        initial_state = np.random.choice([0, 1], size=(100, 100))

        # Use EEG data to modify the rules of the automaton
        rule = int(self.data[0] * 255)  # Convert band power to a rule number

        for step in range(100):  # Run for 100 iterations
            new_state = np.zeros_like(initial_state)
            for i in range(1, 99):
                for j in range(1, 99):
                    # Consider the cell and its immediate neighbors
                    neighborhood = initial_state[i-1:i+2, j-1:j+2]
                    # Apply a rule based on the neighborhood and EEG data
                    new_state[i, j] = 1 if np.sum(neighborhood) > 4 + rule % 5 else 0
            initial_state = new_state

        # Update the image data
        self.img.set_data(initial_state)
        plt.draw()
        plt.pause(0.01)  # pause a bit so that the plot gets updated
    def fractal(self):
        # Set the size of the image (width, height)
        width = 100
        height = 100

        # Create a blank image
        image = np.zeros((width, height))

        # Define properties of the fractal
        x_min, x_max = -2.0, 1.0
        y_min, y_max = -1.5, 1.5
        max_iter = int(max(self.data) * 1000)  # Use band power data here

        # Generate the fractal
        for x in range(width):
            for y in range(height):
                zx, zy = (
                    x * (x_max - x_min) / (width - 1)
                    + x_min
                    + self.data[0],  # Add band power data here
                    y * (y_max - y_min) / (height - 1)
                    + y_min
                    + self.data[1],  # Add band power data here
                )
                c = zx + zy * 1j
                z = c
                for i in range(max_iter):
                    if abs(z) > 2.0:
                        break
                    z = z * z + c
                image[x, y] = i

        # Update the image data instead of creating a new plot
        self.img.set_data(image)
        plt.draw()
        plt.pause(0.01)  # pause a bit so that the plot gets updated

    # def sound_synthesis(self):
    #     # Convert band power data to frequency and duration
    #     frequency = int(self.data[0] * 1000)  # Frequency in Hz
    #     duration = int(self.data[1] * 1000)  # Duration in milliseconds

    #     # Generate a sine wave tone with the given frequency and duration
    #     sample_rate = 44100
    #     t = np.linspace(0, duration / 1000, int(sample_rate * duration / 1000), False)
    #     tone = np.sin(frequency * t * 2 * np.pi)

    #     # Play the tone
    #     sd.play(tone, sample_rate)
    def sound_synthesis(self):
        # Convert band power data to pitch and duration
        pitch = int(self.data[0] * 60) + 60  # MIDI note number
        duration = self.data[1] * 2  # Duration in beats

        # Create a new MIDI output port
        outport = mido.open_output()

        # Send a note on message
        outport.send(Message("note_on", note=pitch, velocity=64))

        # Wait for the duration of the note
        time.sleep(duration)

        # Send a note off message
        outport.send(Message("note_off", note=pitch, velocity=64))

    def interactive_art(self):
        # Interactive art code here
        pass

    def performance_art(self):
        # Performance art code here
        pass

    def diffusion_model(self):
        # Set the size of the image (width, height)
        width = 100
        height = 100

        # Create a blank image
        image = np.zeros((width, height))

        # Use EEG data to set the diffusion rate
        diffusion_rate = self.data[0] * 0.1  # Adjust diffusion rate based on EEG

        # Initialize the diffusion model
        model = np.random.choice([0, 1], size=(width, height), p=[1-diffusion_rate, diffusion_rate])

        # Run the diffusion model
        for _ in range(int(self.data[1] * 100)):  # Use EEG data to determine the number of iterations
            new_model = np.zeros_like(model)
            for i in range(1, width-1):
                for j in range(1, height-1):
                    # Sum the states of the neighboring cells
                    total = np.sum(model[i-1:i+2, j-1:j+2]) - model[i, j]
                    # Apply diffusion rules based on the total
                    new_model[i, j] = 1 if total > 4 else 0
            model = new_model

        # Update the image data
        self.img.set_data(model)
        plt.draw()
        plt.pause(0.01)  # pause a bit so that the plot gets updated

if __name__ == "__main__":
    print("Looking for an EEG stream...")
    streams = resolve_byprop("type", "EEG", timeout=2)
    if len(streams) == 0:
        raise RuntimeError("Can't find EEG stream.")

    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    info = inlet.info()
    description = info.desc()

    fs = int(info.nominal_srate())

    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None

    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) / SHIFT_LENGTH + 1))

    band_buffer = np.zeros((n_win_test, 4))

    print("Press Ctrl-C in the console to break the while loop.")

    # Create the figure and the image object outside the while loop
    fig, ax = plt.subplots()
    img = ax.imshow(
        np.zeros((100, 100)), cmap="Greys", interpolation="nearest", vmin=0, vmax=1
    )
    plt.colorbar(img, ax=ax)  # Add a colorbar
    plt.show(block=False)
    try:
        while True:
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=int(SHIFT_LENGTH * fs)
            )

            ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

            eeg_buffer, filter_state = utils.update_buffer(
                eeg_buffer, ch_data, notch=True, filter_state=filter_state
            )

            data_epoch = utils.get_last_data(eeg_buffer, EPOCH_LENGTH * fs)

            band_powers = utils.compute_band_powers(data_epoch, fs)
            band_buffer, _ = utils.update_buffer(band_buffer, np.asarray([band_powers]))

            smooth_band_powers = np.mean(band_buffer, axis=0)

            alpha_metric = (
                smooth_band_powers[Band.Alpha] / smooth_band_powers[Band.Delta]
            )
            print("Alpha Relaxation: ", alpha_metric)

            # Sum band powers into two groups
            band_powers = [sum(smooth_band_powers[:2]), sum(smooth_band_powers[2:])]

            # Take absolute values to ensure non-negativity
            band_powers = [abs(i) for i in band_powers]

            # Normalize band powers to sum to 1
            band_powers = [float(i) / sum(band_powers) for i in band_powers]
            print(band_powers)
            # Create an instance of ArtGenerator
            art = ArtGenerator(band_powers, img)

            # Generate art using cellular automaton
            art.cellular_automaton()
            # art.fractal()
            # art.diffusion_model()
            # art.sound_synthesis()
    except KeyboardInterrupt:
        sd.stop()
        print("Closing!")
