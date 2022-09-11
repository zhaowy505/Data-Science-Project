# coding: utf-8
from ui.main import Ui_Form
from PyQt5 import QtWidgets, QtGui, QtCore
from plugin_canvas import Canvas
from utils import AbstractFunction
import config
import pandas as pd
import json
from base64 import b64encode
from explanation3 import *


class MainApp(QtWidgets.QWidget, Ui_Form, AbstractFunction):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)
        self.widget_canvas = Canvas(self.frame)
        layout = QtWidgets.QGridLayout(self.frame)
        layout.setContentsMargins(0, 0, 0, 0)
        self.frame.setLayout(layout)
        layout.addWidget(self.widget_canvas, 0, 0, 1, 1)

        self.setWindowTitle(config.app_name)
        self.label_header.setText(config.app_name)

        self.widget_canvas.load_html_file('./static/plot.html')
        self.btnSelectData.clicked.connect(self.confirm_num_and_plot)

        self.data_df = None
        self.btnLoad.clicked.connect(self.load_data)  # button for loading global explanation

        self.table_content.setMidLineWidth(400)  # minimum table width

        self.btnSaveData.clicked.connect(self.read_data_from_table_and_plot_again)  # get the revised data and updated plot

        self.btnSelectData.setDisabled(True)
        self.btnSaveData.setDisabled(True)

        self.table_content.cellDoubleClicked.connect(self.handle_table_db_clicked)  # double click the table


    def load_data(self):
        """
        loading data for global explanation
        """
        self.data_df = pd.read_csv('./data/Employee.csv')  # load original data
        self.data_df = pro_data(self.data_df, random_state=42)  # data processing

        Xtr, Xtest, Ytr, Ytest = data_process(self.data_df, random_state=42)

        model = model_train(Xtr, Ytr, 5)
        is_leaves = leaf_node(model)
        features_name = Xtr.columns.tolist()

        # global explanation
        data = tree_visual(visual_pattern="Global_Explanation", decision_tree=model, is_leaves=leaf_node(model),
                           feature_names=features_name)

        self.update_plot(data=data)

        self.spin_num.setMaximum(self.data_df.shape[0])  # set max value of n，minimal n shoule be 1
        self.init_table(columns=self.data_df.columns)  # Initialization the table

        self.btnSelectData.setDisabled(False)
        self.btnSaveData.setDisabled(False)
        self.btnLoad.setDisabled(True) 

    def confirm_num_and_plot(self,num):
        """
        process the selected data and show its local explanation
        """
        num = self.spin_num.value()
        r = self.show_info_message(message='The number of instance you selected is {}'.format(num))
        if r:
            Xtr, Xtest, Ytr, Ytest = data_process(self.data_df, random_state=42)
            model = model_train(Xtr, Ytr, 5)
            is_leaves = leaf_node(model)
            features_name = Xtr.columns.tolist()

            self.update_table(Xtest.iloc[:num, :].values)
            sample, pred_label = sample_path(model, Xtest, num)
            instance = sample[num - 1]  # load the last data of local explanation
            data = tree_visual(visual_pattern="Local_Explanation", decision_tree=model, is_leaves=leaf_node(model),
                               feature_names=features_name, sample_node_path=instance)
            self.update_plot(data=data)

    def handle_table_db_clicked(self, r, c):
        """
        View the visualization of local explanation by double clicking the instance
        """
        num=r
        Xtr, Xtest, Ytr, Ytest = data_process(self.data_df, random_state=42)
        model = model_train(Xtr, Ytr, 5)
        is_leaves = leaf_node(model)
        features_name = Xtr.columns.tolist()

        sample, pred_label = sample_path(model, Xtest, num+1)
        instance = sample[num]  
        data = tree_visual(visual_pattern="Local_Explanation", decision_tree=model, is_leaves=leaf_node(model),
                               feature_names=features_name, sample_node_path=instance)
        self.update_plot(data=data)

    def init_table(self, columns):
        """
        Initialization the table
        """
        self.table_content.setColumnCount(len(columns))
        self.table_content.setRowCount(0)
        self.table_content.setHorizontalHeaderLabels(list(columns))

    def update_table(self, data):
        """
        updated the data of table, and the data in the table is editable
        """
        self.table_content.setRowCount(0)
        for r, cols in enumerate(data):
            self.table_content.insertRow(r)
            for c, v in enumerate(cols):
                item = QtWidgets.QTableWidgetItem()
                item.setText(str(v))
                self.table_content.setItem(r, c, item)

    def read_data_from_table_and_plot_again(self):

        Xtr, Xtest, Ytr, Ytest = data_process(self.data_df, random_state=42)
        model = model_train(Xtr, Ytr, 5)
        is_leaves = leaf_node(model)
        features_name = Xtr.columns.tolist()
        """
        read data from table and updated plot of counterfactual_explanation
        """
        data = []
        rows = self.table_content.rowCount()
        cols = self.table_content.columnCount()

        for r in range(rows):  # each row
            row_data = []
            for c in range(cols):  # each column
                item: QtWidgets.QTableWidgetItem = self.table_content.item(r, c)
                if item == None:
                    continue
                row_data.append(item.text().strip())

            data.append(row_data)

        # updated the last data and updated plot
        num = self.spin_num.value()
        initial_data = Xtest.head(num).iloc[[num-1]]
        print(data)
        data = data[num - 1][:8]
        data = np.array(data).reshape(1, 8)
        modified_data = pd.DataFrame(data, columns=features_name)  # transformation to Dataframe

        print(initial_data)
        print(modified_data)

        json_data = tree_visual(visual_pattern="Counterfactual_Explanation", decision_tree=model,
                                is_leaves=leaf_node(model), feature_names=features_name, initial=initial_data,
                                modified=modified_data)
        self.update_plot(data=json_data)

    def update_plot(self, data: [dict, list]):
        """
        :param data:
        :return:
        """
        json_encoded_data = json.dumps(data)
        base64_encoded_data = b64encode(json_encoded_data.encode('utf-8')).decode('utf-8')
        self.widget_canvas.page().runJavaScript('updateChart("%s")' % base64_encoded_data)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    m = MainApp()
    m.show()
    app.exec()
