import numpy as np
import serial
import sys
import time
import tkinter as tk
from PIL import Image, ImageTk
from sklearn.metrics.pairwise import haversine_distances

EARTH_RADIUS = 6371
PROCESSED_DATA_DIR = "/home/kaptim/eth/mlmc/project/bottom_up/code/processed_data/"


def haversine_distance_np(y_true, y_pred):
    # expects numpy input in (latitude, longitude) format: N x 2
    # reshape if a single dimension
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
    y_true_radians = np.radians(y_true)
    y_pred_radians = np.radians(y_pred)

    distance = np.diag(haversine_distances(y_true_radians, y_pred_radians))

    return EARTH_RADIUS * distance


def main():
    x_test = np.load(PROCESSED_DATA_DIR + "x_test_CH_96_96_q.npy")
    y_test = np.load(PROCESSED_DATA_DIR + "y_test_CH_96_96_q.npy")

    print(f"Loaded x with shape: {x_test.shape}")
    print(f"Loaded y with shape: {y_test.shape}")

    ser = serial.Serial(port="COM3", baudrate=115200, timeout=3)
    # flush the serial port
    ser.flush()
    ser.flushInput()
    ser.flushOutput()

    distance_sum = 0
    # define how many images from the test set to send to the MCU
    test_len = 20
    # get how many predictions we iterated over
    num_pred = 0

    for x, y in zip(x_test[:test_len], y_test[:test_len]):
        num_pred += 1
        ser.write(x.tobytes())
        time.sleep(1)
        img = ser.read(x.shape[0] * x.shape[1])
        img = np.frombuffer(img, dtype=np.uint8)
        print("Image sent to the MCUs: \n {}".format(img))
        time.sleep(1)
        pred = ser.read(y.shape[0])
        pred = np.frombuffer(pred, dtype=np.uint8)
        print(f"Target: {y}, Prediction (from MCU): {pred}")
        distance_sum += haversine_distance_np(y, pred)

        root = tk.Tk()
        root.title("Real-time Inference")

        # Create an image from the numpy array
        image = Image.fromarray(x.astype(np.uint8))
        image = image.resize((280, 280))  # Adjust size as needed
        photo = ImageTk.PhotoImage(image)

        root.grid_rowconfigure((0, 1, 2, 3, 4), weight=1)
        root.grid_columnconfigure((0, 1), weight=1)
        bg_color = "#FFFFFF"
        darkgrey = "#2D2D2D"
        yellow = "#FFC107"
        quit_color = "#DA0037"
        next_color = "#3E497A"
        next_hover_color = "#3AB4F2"

        # Display the image
        image_label = tk.Label(master=root, image=photo)
        image_label.image = photo  # Keep a reference to prevent garbage collection
        image_label.grid(row=0, column=0, padx=(10, 5), pady=10, columnspan=2)

        # Prediction label
        pred_text = f"Prediction: {pred}"
        pred_label = tk.Label(
            root,
            text=pred_text,
            font=("Helvetica", 18, "bold"),
            bg=bg_color,
            fg="white",
        )
        pred_label.grid(row=1, column=0, padx=(10, 5), pady=10, sticky="nsew")

        # Target label
        target_text = f"Target: {y}"
        target_label = tk.Label(
            root,
            text=target_text,
            font=("Helvetica", 18, "bold"),
            bg=bg_color,
            fg="white",
        )
        target_label.grid(row=1, column=1, padx=(5, 10), pady=10, sticky="nsew")

        # Average haversine distance
        accuracy = distance_sum / num_pred
        accuracy_text = f"Average haversine distance: {accuracy:.2f}%"
        accuracy_label = tk.Label(
            root,
            text=accuracy_text,
            font=("Helvetica", 18, "bold"),
            bg=yellow,
            fg=darkgrey,
        )
        accuracy_label.grid(
            row=2, column=0, padx=(10, 10), pady=10, columnspan=2, sticky="nsew"
        )

        # Functions to handle button clicks
        def next_image():
            root.quit()
            root.destroy()

        def quit_program():
            root.quit()
            root.destroy()
            sys.exit()

        # Next button
        next_button = tk.Button(
            root,
            text="Next",
            command=next_image,
            font=("Helvetica", 18, "bold"),
            bg=next_color,
            fg="white",
            activebackground=next_hover_color,
            activeforeground="white",
        )
        next_button.grid(
            row=3, column=0, padx=(10, 5), pady=10, columnspan=2, sticky="nsew"
        )

        # Quit button
        quit_button = tk.Button(
            root,
            text="Quit",
            command=quit_program,
            font=("Helvetica", 18, "bold"),
            bg=next_color,
            fg="white",
            activebackground=quit_color,
            activeforeground="white",
        )
        quit_button.grid(
            row=4, column=0, padx=(10, 5), pady=10, columnspan=2, sticky="nsew"
        )

        root.mainloop()


if __name__ == "__main__":
    main()
