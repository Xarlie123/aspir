from PyQt5.QtWidgets import QFileDialog

def seleccionar_archivo(line_edit):
    """Opens a dialog to select a file and displays the path in the QLineEdit."""
    archivo, _ = QFileDialog.getOpenFileName(None, "Select image", "", "Images (*.png *.jpg *.jpeg *.bmp);;All files (*.*)")
    if archivo:
        line_edit.setText(archivo)

def seleccionar_directorio(line_edit):
    """Opens a dialog to select a folder and displays the path in the QLineEdit."""
    carpeta = QFileDialog.getExistingDirectory(None, "Select folder")
    if carpeta:
        line_edit.setText(carpeta)
