import sys
import cv2
import numpy as np
import joblib
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
from deepface import DeepFace
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QFileDialog
from PyQt5.QtCore import QTimer, Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon

common_stylesheet = """
    color: #C71585;                 /* Text color */
    border: 2px solid #C8105A;     /* Border color and width */
    padding: 15px 20px;            /* Top-Bottom | Left-Right padding */
    background-color: #FDEEF4;     /* Background color for better visibility */
    font-size: 30px;               /* Font size for consistent heading size */
    font-weight: bold;             /* Font weight for bold text */
"""

class EmotionRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.is_recording = False
        self.audio_stream = None
        # Load cry recognition model and label encoder
        self.cry_model = joblib.load('cry_recognition_model.pkl')
        self.label_encoder = joblib.load('label_encoder.pkl')

    def initUI(self):
        self.setWindowTitle("BabyMood")
        self.setGeometry(100, 100, 1200, 900)
        self.setStyleSheet("background-color: #FFB6C1;")

        self.title_label = QLabel("Emotion Detection", self)
        self.title_label.setFont(QFont('Arial', 28, QFont.Bold))
        self.title_label.setStyleSheet(common_stylesheet)
        self.title_label.setAlignment(Qt.AlignCenter)
        
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(800, 600) 
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 5px solid #FF66CC; border-radius: 15px; background-color: #2C3E50;")

        self.emotion_label = QLabel("Detected Emotion: None", self)
        self.emotion_label.setFont(QFont('Arial', 20))
        self.emotion_label.setStyleSheet("color: #3B2A0A; margin-top: 20px;")
        self.emotion_label.setAlignment(Qt.AlignCenter)

        self.cry_title_label = QLabel("Cry Classification", self)
        self.cry_title_label.setFont(QFont('Arial', 24, QFont.Bold))
        self.cry_title_label.setStyleSheet(common_stylesheet)
        self.cry_title_label.setAlignment(Qt.AlignCenter)
        self.cry_title_label.setFixedWidth(600)

        self.cry_result_label = QLabel("Cry Classification Result: None", self)
        self.cry_result_label.setFont(QFont('Arial', 20))
        self.cry_result_label.setStyleSheet("color: #3B2A0A; margin-top: 20px;")
        self.cry_result_label.setAlignment(Qt.AlignCenter)

        self.upload_audio_button = QPushButton("Upload Audio", self)
        self.upload_audio_button.setIcon(QIcon())
        self.upload_audio_button.setIconSize(QSize(24, 24))
        self.upload_audio_button.setStyleSheet(self.button_style())
        self.upload_audio_button.clicked.connect(self.upload_audio)

        # New button for recording audio
        self.record_audio_button = QPushButton("Record Audio", self)
        self.record_audio_button.setIcon(QIcon())
        self.record_audio_button.setIconSize(QSize(24, 24))
        self.record_audio_button.setStyleSheet(self.button_style())
        self.record_audio_button.clicked.connect(self.toggle_recording)

        self.start_button = QPushButton("Start", self)
        self.start_button.setIcon(QIcon("start_icon.png"))
        self.start_button.setIconSize(QSize(24, 24))
        self.start_button.setStyleSheet(self.button_style())
        self.start_button.clicked.connect(self.start_video)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.setIcon(QIcon("stop_icon.png"))
        self.stop_button.setIconSize(QSize(24, 24))
        self.stop_button.setStyleSheet(self.button_style())
        self.stop_button.clicked.connect(self.stop_video)

        self.quit_button = QPushButton("Quit", self)
        self.quit_button.setIcon(QIcon("quit_icon.png"))
        self.quit_button.setIconSize(QSize(24, 24))
        self.quit_button.setStyleSheet(self.button_style())
        self.quit_button.clicked.connect(self.quit_app)

        self.upload_image_button = QPushButton("Upload Image", self)
        self.upload_image_button.setIcon(QIcon("image_icon.png"))
        self.upload_image_button.setIconSize(QSize(24, 24))
        self.upload_image_button.setStyleSheet(self.button_style())
        self.upload_image_button.clicked.connect(self.upload_image)

        self.upload_video_button = QPushButton("Upload Video", self)
        self.upload_video_button.setIcon(QIcon("video_icon.png"))
        self.upload_video_button.setIconSize(QSize(24, 24))
        self.upload_video_button.setStyleSheet(self.button_style())
        self.upload_video_button.clicked.connect(self.upload_video)

        self.play_audio_button = QPushButton("Play Audio", self)
        self.play_audio_button.setIcon(QIcon("play_icon.png"))
        self.play_audio_button.setIconSize(QSize(24, 24))
        self.play_audio_button.setStyleSheet(self.button_style())
        self.play_audio_button.clicked.connect(self.play_audio)

        # New button to stop audio playback
        self.stop_audio_button = QPushButton("Stop Audio", self)
        self.stop_audio_button.setIcon(QIcon("stop_audio_icon.png"))
        self.stop_audio_button.setIconSize(QSize(24, 24))
        self.stop_audio_button.setStyleSheet(self.button_style())
        self.stop_audio_button.clicked.connect(self.stop_audio)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.upload_image_button)
        button_layout.addWidget(self.upload_video_button)
        button_layout.addWidget(self.quit_button)
        button_layout.setAlignment(Qt.AlignCenter)
    
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.title_label)

        camera_widget = QWidget()
        camera_layout = QVBoxLayout(camera_widget)
        camera_layout.addWidget(self.image_label)
        camera_layout.setAlignment(Qt.AlignCenter)
        camera_widget.setLayout(camera_layout)
    
        main_layout.addWidget(camera_widget)
        main_layout.addWidget(self.emotion_label)
        main_layout.addLayout(button_layout)

        cry_section = QWidget()
        cry_layout = QVBoxLayout(cry_section)
        cry_layout.addWidget(self.cry_title_label)
        cry_layout.addWidget(self.cry_result_label)
        cry_layout.addWidget(self.upload_audio_button)
        cry_layout.addWidget(self.record_audio_button)  # Added record audio button here
        cry_layout.addWidget(self.play_audio_button)
        cry_layout.addWidget(self.stop_audio_button)    # Added stop audio button here
        cry_layout.setAlignment(Qt.AlignCenter)
        cry_section.setLayout(cry_layout)

        main_layout.addWidget(cry_section)

        self.setLayout(main_layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
    
        self.cap = None
        self.video_cap = None
        self.video_file_path = None
        self.recording_file = None
        self.audio_data = None

    def button_style(self):
        return """
        QPushButton {
            background-color: #2980B9;
            color: white;
            border-radius: 15px;
            padding: 10px 20px;
            font-size: 16px;
        }
        QPushButton:hover {
            background-color: #3498DB;
        }
        QPushButton:pressed {
            background-color: #1F618D;
        }
        QPushButton:disabled {
            background-color: #95A5A6;
        }
        """

    def toggle_recording(self):
        if not self.is_recording:
            self.record_audio_button.setText("Stop Recording")
            self.start_recording()
        else:
            self.record_audio_button.setText("Record Audio")
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.fs = 44100  # Sample rate
        self.duration = 5  # Duration in seconds
        self.audio_data = sd.rec(int(self.duration * self.fs), samplerate=self.fs, channels=1)
        sd.wait()  # Wait until recording is finished

    def stop_recording(self):
        self.is_recording = False
        file_name = "recorded_audio.wav"
        write(file_name, self.fs, self.audio_data)  # Save as WAV file
        self.process_audio(file_name)

    def start_video(self):
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)

    def stop_video(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
            self.image_label.clear()
            self.emotion_label.setText("Detected Emotion: None")

        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None

    def update_frame(self):
        if self.video_cap:
            ret, frame = self.video_cap.read()
            if not ret:
                self.stop_video()
                return
        else:
            ret, frame = self.cap.read()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame_rgb.shape
            q_img = QImage(frame_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_img))
            self.detect_emotion(frame)

    def detect_emotion(self, frame):
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            self.emotion_label.setText(f"Detected Emotion: {emotion}")
        except Exception as e:
            self.emotion_label.setText("Emotion detection error")

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            image = QImage(file_name)
            if image.isNull():
                self.image_label.setText("Failed to load image")
                return
            self.image_label.setPixmap(QPixmap.fromImage(image))
            frame = cv2.imread(file_name)
            self.detect_emotion(frame)

    def upload_video(self):
        self.video_file_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if self.video_file_path:
            self.video_cap = cv2.VideoCapture(self.video_file_path)
            self.timer.start(30)

    def upload_audio(self):
        self.recording_file, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "Audio Files (*.wav)")
        if self.recording_file:
            self.process_audio(self.recording_file)

    def process_audio(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=None)
        
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfccs_processed = np.mean(mfccs.T, axis=0).reshape(1, -1)
        
            # Predict using the model
            prediction = self.cry_model.predict(mfccs_processed)
            predicted_label = self.label_encoder.inverse_transform(prediction)
            self.cry_result_label.setText(f"Cry Classification Result: {predicted_label[0]}")
        
        except Exception as e:
                print(f"Error processing audio: {e}")
                self.cry_result_label.setText("Error processing audio")


    def play_audio(self):
        if self.recording_file:
            audio_data, fs = librosa.load(self.recording_file, sr=None)
            self.audio_stream = sd.OutputStream(samplerate=fs, channels=1, dtype='float32')
            self.audio_stream.start()
            self.audio_stream.write(audio_data)
        
    def stop_audio(self):
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None

    def quit_app(self):
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EmotionRecognitionApp()
    window.show()
    sys.exit(app.exec_())
