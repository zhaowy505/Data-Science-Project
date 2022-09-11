# coding: utf-8
# coding: utf-8
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings, QWebEnginePage
from PyQt5.QtWebChannel import QWebChannel
import inspect
import typing
import os
import config


class QMyWebEnginePage(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        print("javaScriptConsoleMessage: ", level, message, lineNumber, sourceID)


class Canvas(QWebEngineView):
    sig_who_db_clicked = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPage(QMyWebEnginePage(self))
        self.is_loaded = False
        self.loadFinished.connect(self.handle_loaded)
        self.run_after_load_finished_callback = None

        self.channel = QWebChannel()
        self.channel.registerObject('pyHandle', self)
        self.page().setWebChannel(self.channel)

        self.settings().setAttribute(QWebEngineSettings.LocalStorageEnabled, True) 
        self.setContextMenuPolicy(QtCore.Qt.NoContextMenu) 
    def load_html_file(self, fp: str):
        """

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
<div id="main">Loading...</div>
</body>
</html>
        """
        if config.DEBUG is True:
            if not os.path.exists(fp):
                raise FileNotFoundError
            fp = os.path.abspath(fp).replace('\\', '/')
            self.load(QtCore.QUrl(fp))
        else:
            if fp.startswith('./static'):
                self.load(QtCore.QUrl(f'qrc:{fp[1:]}'))
            else:
                if not os.path.exists(fp):
                    raise FileNotFoundError
                fp = os.path.abspath(fp).replace('\\', '/')
                self.load(QtCore.QUrl(fp))

    def load_html_content(self, content: str):
        """

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
<div id="main">Loading...</div>
</body>
</html>
        """
        self.setHtml(content)

    def handle_loaded(self):
        self.is_loaded = True
        if self.run_after_load_finished_callback is not None:
            self.run_after_load_finished_callback()

    @QtCore.pyqtSlot(str)
    def dbClicked(self, name: str):
        self.sig_who_db_clicked.emit(name)

    def apply_dark_theme(self):
        self.page().setBackgroundColor(QtGui.QColor('#090C39'))

    def apply_light_theme(self):
        self.page().setBackgroundColor(QtGui.QColor('#ffffff'))

