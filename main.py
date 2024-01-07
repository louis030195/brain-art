# -*- coding: utf-8 -*-
"""
Estimate Relaxation from Band Powers

Run:
muselsl stream
python3 main.py

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylsl import StreamInlet, resolve_byprop
import utils


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


def step(state):
    print("State shape in step:", state.shape)
    num_neighbors = sum(
        np.roll(np.roll(state, i, 0), j, 1)
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        if (i != 0 or j != 0)
    )
    return (num_neighbors == 3) | ((state.astype(int) & (num_neighbors == 2)) != 0)


class ArtGenerator:
    def __init__(self, data, img):
        self.data = data
        self.img = img

    def cellular_automaton(self):
        # Create a random initial state
        initial_state = np.random.choice([0, 1], size=(100, 100))
        # Set the probabilities
        initial_state = (
            initial_state * self.data[0] + (1 - initial_state) * self.data[1]
        )

        # Ensure initial_state is a 2-dimensional array
        initial_state = np.reshape(initial_state, (100, 100))

        # Update the image data instead of creating a new plot
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
                    x * (x_max - x_min) / (width - 1) + x_min,
                    y * (y_max - y_min) / (height - 1) + y_min,
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

    def sound_synthesis(self):
        # Sound synthesis code here
        pass

    def interactive_art(self):
        # Interactive art code here
        pass

    def performance_art(self):
        # Performance art code here
        pass


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

            # Create an instance of ArtGenerator
            art = ArtGenerator(band_powers, img)

            # Generate art using cellular automaton
            # art.cellular_automaton()
            art.fractal()
    except KeyboardInterrupt:
        print("Closing!")
