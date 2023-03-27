import os
import cv2
import numpy as np
from mhyt import yt_download
from moviepy.editor import *
from collections import deque
import time
from keras.models import load_model
import tkinter as tk
from tkinter.ttk import *
from PIL import ImageTk, Image
from tkvideo import tkvideo

# load model
model = load_model(
    'Model_3____Loss_0.10751260817050934___Accuracy_0.970588207244873.h5')

categories = ['arabesque', 'grand-jete', 'pirouette', 'pa-de-bourree']


# Creating The Output directories if it does not exist
output_directory = 'user_videos'
os.makedirs(output_directory, exist_ok=True)

img_size = 200

# Setting the Window Size which will be used by the Rolling Average Process
window_size = 20  # 25


# Creating App class

class App:

    def __init__(self, master) -> None:
        # Instantiating master i.e toplevel Widget
        self.master = master

        # Upper frame
        frm_up = tk.Frame(self.master, bg="black")
        frm_up.pack()

        # Middle frame
        frm_mid = tk.Frame(self.master, bg="black")
        frm_mid.pack(fill="both")

        frm_mid.columnconfigure(0, weight=1)
        frm_mid.columnconfigure(1, weight=2)
        frm_mid.columnconfigure(2, weight=1)

        # Bottom frame
        frm_bt = tk.Frame(self.master, bg="black")
        # frm_up.grid(column=0, row=0)
        frm_bt.pack(fill="both", expand=True)

        # Image
        try:
            img = ImageTk.PhotoImage(Image.open('icon/ballet shoes.png'))
            panel = Label(frm_up, image=img)
            panel.img = img
            panel.grid(column=1, row=0, pady=20)
        except:
            pass

        # Header Label 1
        lbl_main1 = tk.Label(frm_up,
                             anchor="w",
                             text="Estimate",
                             fg="orchid4",
                             bg="black",
                             font=("Arial", 20)
                             )
        lbl_main1.grid(column=0, row=0, padx=10)

        # Header Label2
        lbl_main2 = tk.Label(frm_up,
                             anchor="e",
                             text="Poses",
                             fg="orchid4",
                             bg="black",
                             font=("Arial", 20)
                             )

        lbl_main2.grid(column=2, row=0, padx=10)

        # Instruction Label
        lbl_instr = tk.Label(frm_mid,
                             text="Enter video URL:",
                             fg="seashell4",
                             bg="black",
                             font=("Arial", 10)
                             )
        lbl_instr.grid(column=0, row=0, padx=10)

        # Warning Label
        self.lbl_wrng = tk.Label(frm_mid,
                                 fg="red4",
                                 bg="black",
                                 font=("Arial", 8),
                                 width=40
                                 )
        self.lbl_wrng.grid(column=1, row=1)

        # Text entry
        self.entry = tk.Entry(frm_mid, width=70, bg="seashell3")
        self.entry.grid(column=1, row=0, pady=2)

        # Button
        button = tk.Button(frm_mid,
                           text="Start",
                           width=10,
                           bg="seashell4",
                           fg="black")
        button.grid(column=2, row=0, padx=8, pady=8)
        button.bind('<Button-1>', self.handle_click)

        # Video Label
        self.lbl_vid = tk.Label(frm_bt, bg="black")
        self.lbl_vid.pack()

    def download_youtube_videos(self, youtube_video_url, output_directory):
        download_success = False
        try:
            # Set label
            self.lbl_wrng['text'] = "Loading.... Please wait a few minutes!"

            title = 'video_to_predict'

            # Constructing the Output File Path
            output_file_path = f'{output_directory}/{title}.mp4'

            # Download youtube video
            yt_download(youtube_video_url, output_file_path)

            download_success = True

        except:
            self.lbl_wrng['text'] = "Video URL is not valid"
        finally:
            self.entry.delete(0, tk.END)

        # Returning Video Title
        return download_success

    def predict_on_video(self, video_file_path, output_file_path, window_size):
        # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
        predicted_labels_probabilities_deque = deque(maxlen=window_size)

        # Reading the Video File using the VideoCapture Object
        video_reader = cv2.VideoCapture(video_file_path)

        # Getting the width and height of the video
        original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_video_height = int(
            video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_video_fps = video_reader.get(cv2.CAP_PROP_FPS)

        # Writing the Overlayed Video Files Using the VideoWriter Object
        video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(
            *'mp4v'), original_video_fps, (original_video_width, original_video_height))

        while True:
            # Reading The Frame
            status, frame = video_reader.read()
            if not status:
                break
            # Resize the Frame to fixed Dimensions
            resized_frame = cv2.resize(frame, (img_size, img_size))
            # Convert the frame to grayscale
            grayscale_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2GRAY)
            # Reshape the Frame
            reshaped_frame = grayscale_frame.reshape(-1, img_size, img_size, 1)
            # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
            normalized_frame = reshaped_frame / 255
            # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
            predicted_labels_probabilities = model.predict(normalized_frame)[0]
            # Appending predicted label probabilities to the deque object
            predicted_labels_probabilities_deque.append(
                predicted_labels_probabilities)
            # Assuring that the Deque is completely filled before starting the averaging process
            if len(predicted_labels_probabilities_deque) == window_size:
                # Converting Predicted Labels Probabilities Deque into Numpy array
                predicted_labels_probabilities_np = np.array(
                    predicted_labels_probabilities_deque)
                # Calculating Average of Predicted Labels Probabilities Column Wise
                predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(
                    axis=0)
                # Converting the predicted probabilities into labels by returning the index of the maximum value.
                predicted_label = np.argmax(
                    predicted_labels_probabilities_averaged)
                # Accessing The Class Name using predicted label.
                predicted_class_name = categories[predicted_label]
                # Overlaying Class Name Text Ontop of the Frame
                cv2.putText(frame, predicted_class_name, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (160, 160,160), 2)
            # Writing The Frame
            video_writer.write(frame)

        # Closing the VideoCapture and VideoWriter objects and releasing all resources held by them.
        video_reader.release()
        video_writer.release()

    def handle_click(self, event):
        # Clear Label
        self.lbl_wrng['text'] = ""

        if len(self.entry.get()) == 0:
            self.lbl_wrng['text'] = "Enter video URL first"
        else:
            # Delete old videos from file
            self.delete_vid()

            # Get video URL
            video_url = self.entry.get()

            # Downloading a YouTube Video
            download_success = self.download_youtube_videos(
                video_url, output_directory)

            video_title = 'video_to_predict'

            if download_success == True:
                # Getting the YouTube Video's path you just downloaded
                input_video_file_path = f'{output_directory}/{video_title}.mp4'
                # Constructing The Output YouTube Video Path
                output_video_file_path = f'{output_directory}/predicted_video -Output-WSize {window_size}.mp4'

                # wait for download to complete
                wait = True
                while(wait == True):
                    for title in os.listdir(output_directory):
                        if (video_title) not in title:
                            time.sleep(60)
                        else:
                            wait = False

                # Calling the predict_on_live_video method to start the Prediction and Rolling Average Process
                self.predict_on_video(
                    input_video_file_path, output_video_file_path, window_size)

                # Clear loading Label
                self.lbl_wrng['text'] = ""

                # Play video
                player = tkvideo(output_video_file_path,
                                 self.lbl_vid, loop=0, size=(600, 400))
                player.play()

    def delete_vid(self):
        for f in os.listdir(output_directory):
            os.remove(os.path.join(output_directory, f))


if __name__ == "__main__":

    # Instantiating top level
    root = tk.Tk()

    # Setting the title of the window
    root.title("Ballet Pose Estimation")

    # Setting background color
    root.configure(background='black')

    # Setting the geometry i.e Dimensions
    root.geometry('810x630')

    # Calling our App
    app = App(root)

    # Mainloop
    root.mainloop()
