import sys
import os
import torch
from torchvision import transforms
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QMessageBox, QGraphicsView, QGraphicsScene, QLabel
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QPoint
from PIL import Image, ImageOps
from classifier import CNNClassifier
import matplotlib
from matplotlib import pyplot as plt
# DrawingCanvas class for capturing drawings
class DrawingCanvas(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set up the drawing area size
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)

        # Create and set up the QGraphicsScene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setSceneRect(0, 0, 640, 640)  # Set canvas size

        # Create a QPixmap to draw on (off-screen drawing surface)
        self.canvas_pixmap = QPixmap(480, 480)
        self.canvas_pixmap.fill(Qt.white)  # Set the background to white

        # Add QPixmap to the scene as an item
        self.pixmap_item = self.scene.addPixmap(self.canvas_pixmap)

        # Set the current drawing state and previous point
        self.drawing = False
        self.last_point = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing:
            painter = QPainter(self.canvas_pixmap)  # Draw directly on the QPixmap
            pen = QPen(QColor(0, 0, 0), 17)  # Black color with a 5px width
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()

            # Since the pixmap has changed, update the scene and view
            self.pixmap_item.setPixmap(self.canvas_pixmap)  # Update the pixmap in the scene
            self.scene.update()  # Request a re-render of the scene

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def keyPressEvent(self, event):
        """Handle key press events to clear the canvas"""
        if event.key() == Qt.Key_C:  # If 'C' is pressed
            self.clear_canvas()  # Clear the canvas
        super().keyPressEvent(event) 
   
    def clear_canvas(self):
        """Clear the drawing area by filling the pixmap with white"""
        self.canvas_pixmap.fill(Qt.white)
        self.pixmap_item.setPixmap(self.canvas_pixmap)
        self.scene.update()  # Update the scene after clearing
       

    def save_image(self, path="drawing.png"):
        try:
            # Save the current drawing as an image
            if self.canvas_pixmap.save(path):
                print(f"Drawing saved to {path}")
            else:
                raise Exception("Failed to save the image.")
        except Exception as e:
            print(f"Error saving image: {e}")
            raise e  # Re-raise the exception for further handling

# Main window class
class ImageCaptureApp(QMainWindow):
    def __init__(self, mappings):
        super().__init__()

        self.setWindowTitle("Drawing Canvas with Inference")
        self.setGeometry(100, 100, 640, 480)
        self.mappings = mappings

        # Set up GUI components
        self.canvas = DrawingCanvas(self)
        self.capture_button = QPushButton("Predict Character", self)
        self.capture_button.clicked.connect(self.capture_drawing)

        # Add a QLabel for text at the bottom of the canvas
        self.text_label = QLabel("Draw a character that fills the canvas", self)
        self.text_label.setAlignment(Qt.AlignCenter)  # Center the text in the label
        self.text_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #FFF;")  # Styling text
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.text_label)  # Add the text label at the bottom

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
   
    
    def capture_drawing(self):
        try:
            # Save the drawing to an image file
            img_path = "drawing.png"
            self.canvas.save_image(img_path)

            # Run inference on the captured image
            self.run_inference(img_path)
        except Exception as e:
            # Show an error message if any exception occurs
            self.show_error_message(str(e))

    def run_inference(self, image_path):
        try:
            # Check if Metal backend is available (for macOS GPU acceleration)
            device = torch.device("mps" if torch.has_mps else "cpu")
            print(f"Using device: {device}")

            # Check if model file exists
            model = CNNClassifier().to(device)
            print("Model initialized")
            model.load_state_dict(torch.load("model_state_dict.pth"))
            print("Loaded model")
            model.eval()
            
         
            # Load and preprocess the image
            img = Image.open(image_path).convert('L')
            img = img.rotate(-90, expand=True)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = ImageOps.invert(img)
            img = img.resize((28, 28), resample=Image.Resampling.LANCZOS)
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ])
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            img_numpy = img_tensor.squeeze().cpu().numpy()
            # Perform inference
            with torch.no_grad():
                output = model(img_tensor)

            # Post-process the output (e.g., classification)
            _, predicted_class = torch.max(output, 1)
            plt.close()
            plt.imshow(img_numpy, cmap="gray")
            plt.show()
            print(f"Predicted class: {self.mappings[predicted_class.item()]}")
            self.text_label.setText(f"Predicted class: {self.mappings[predicted_class.item()]}. Press 'c' to clear.")
        except Exception as e:
            # Show an error message if any exception occurs
            self.show_error_message(str(e))
    def keyPressEvent(self, event):
        """Handle key press events to clear the canvas"""
        if event.key() == Qt.Key_C:  # If 'C' is pressed
            self.text_label.setText("Draw a character that fills the canvas")

    def show_error_message(self, message):
        """Show an error message in a pop-up dialog."""
        QMessageBox.critical(self, "Error", message)

def main():
    with open("mappings.txt", "r") as f:
        lines = f.readlines()

    mappings = {}

    for line in lines:
        key, value = line.strip().split(' ')
        mappings[int(key.strip())] = chr(int(value.strip()))

    app = QApplication(sys.argv)
    window = ImageCaptureApp(mappings)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
