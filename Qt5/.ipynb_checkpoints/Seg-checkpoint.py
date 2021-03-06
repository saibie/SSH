# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'LungCTSegmenter.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_LungCTSegmenter(object):
    def setupUi(self, LungCTSegmenter):
        LungCTSegmenter.setObjectName("LungCTSegmenter")
        LungCTSegmenter.resize(445, 562)
        self.verticalLayout = QtWidgets.QVBoxLayout(LungCTSegmenter)
        self.verticalLayout.setObjectName("verticalLayout")
        self.inputsCollapsibleButton = ctkCollapsibleButton(LungCTSegmenter)
        self.inputsCollapsibleButton.setObjectName("inputsCollapsibleButton")
        self.formLayout_2 = QtWidgets.QFormLayout(self.inputsCollapsibleButton)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label = QtWidgets.QLabel(self.inputsCollapsibleButton)
        self.label.setObjectName("label")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.inputVolumeSelector = qMRMLNodeComboBox(self.inputsCollapsibleButton)
        self.inputVolumeSelector.setNodeTypes(['vtkMRMLScalarVolumeNode'])
        self.inputVolumeSelector.setShowChildNodeTypes(False)
        self.inputVolumeSelector.setAddEnabled(False)
        self.inputVolumeSelector.setRemoveEnabled(False)
        self.inputVolumeSelector.setObjectName("inputVolumeSelector")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.inputVolumeSelector)
        self.label_2 = QtWidgets.QLabel(self.inputsCollapsibleButton)
        self.label_2.setObjectName("label_2")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.outputSegmentationSelector = qMRMLNodeComboBox(self.inputsCollapsibleButton)
        self.outputSegmentationSelector.setNodeTypes(['vtkMRMLSegmentationNode'])
        self.outputSegmentationSelector.setShowChildNodeTypes(False)
        self.outputSegmentationSelector.setNoneEnabled(True)
        self.outputSegmentationSelector.setAddEnabled(False)
        self.outputSegmentationSelector.setRemoveEnabled(True)
        self.outputSegmentationSelector.setEditEnabled(True)
        self.outputSegmentationSelector.setRenameEnabled(True)
        self.outputSegmentationSelector.setObjectName("outputSegmentationSelector")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.outputSegmentationSelector)
        self.verticalLayout.addWidget(self.inputsCollapsibleButton)
        self.outputsCollapsibleButton = ctkCollapsibleButton(LungCTSegmenter)
        self.outputsCollapsibleButton.setObjectName("outputsCollapsibleButton")
        self.gridLayout = QtWidgets.QGridLayout(self.outputsCollapsibleButton)
        self.gridLayout.setObjectName("gridLayout")
        self.applyButton = QtWidgets.QPushButton(self.outputsCollapsibleButton)
        self.applyButton.setEnabled(False)
        self.applyButton.setObjectName("applyButton")
        self.gridLayout.addWidget(self.applyButton, 5, 1, 1, 1)
        self.cancelButton = QtWidgets.QPushButton(self.outputsCollapsibleButton)
        self.cancelButton.setEnabled(False)
        self.cancelButton.setObjectName("cancelButton")
        self.gridLayout.addWidget(self.cancelButton, 5, 0, 1, 1)
        self.instructionsLabel = ctkFittedTextBrowser(self.outputsCollapsibleButton)
        self.instructionsLabel.setObjectName("instructionsLabel")
        self.gridLayout.addWidget(self.instructionsLabel, 0, 0, 1, 2)
        self.adjustPointsGroupBox = ctkCollapsibleGroupBox(self.outputsCollapsibleButton)
        self.adjustPointsGroupBox.setObjectName("adjustPointsGroupBox")
        self.formLayout_4 = QtWidgets.QFormLayout(self.adjustPointsGroupBox)
        self.formLayout_4.setObjectName("formLayout_4")
        self.label_4 = QtWidgets.QLabel(self.adjustPointsGroupBox)
        self.label_4.setObjectName("label_4")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.rightLungPlaceWidget = qSlicerMarkupsPlaceWidget(self.adjustPointsGroupBox)
        self.rightLungPlaceWidget.setEnabled(True)
        self.rightLungPlaceWidget.setButtonsVisible(True)
        self.rightLungPlaceWidget.setPlaceMultipleMarkups(qSlicerMarkupsPlaceWidget.ForcePlaceMultipleMarkups)
        self.rightLungPlaceWidget.setObjectName("rightLungPlaceWidget")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.rightLungPlaceWidget)
        self.label_5 = QtWidgets.QLabel(self.adjustPointsGroupBox)
        self.label_5.setObjectName("label_5")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.leftLungPlaceWidget = qSlicerMarkupsPlaceWidget(self.adjustPointsGroupBox)
        self.leftLungPlaceWidget.setPlaceMultipleMarkups(qSlicerMarkupsPlaceWidget.ForcePlaceMultipleMarkups)
        self.leftLungPlaceWidget.setObjectName("leftLungPlaceWidget")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.leftLungPlaceWidget)
        self.label_6 = QtWidgets.QLabel(self.adjustPointsGroupBox)
        self.label_6.setObjectName("label_6")
        self.formLayout_4.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.tracheaPlaceWidget = qSlicerMarkupsPlaceWidget(self.adjustPointsGroupBox)
        self.tracheaPlaceWidget.setPlaceMultipleMarkups(qSlicerMarkupsPlaceWidget.ForcePlaceMultipleMarkups)
        self.tracheaPlaceWidget.setObjectName("tracheaPlaceWidget")
        self.formLayout_4.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.tracheaPlaceWidget)
        self.gridLayout.addWidget(self.adjustPointsGroupBox, 7, 0, 1, 2)
        self.toggleSegmentationVisibilityButton = QtWidgets.QPushButton(self.outputsCollapsibleButton)
        self.toggleSegmentationVisibilityButton.setObjectName("toggleSegmentationVisibilityButton")
        self.gridLayout.addWidget(self.toggleSegmentationVisibilityButton, 6, 0, 1, 2)
        self.startButton = QtWidgets.QPushButton(self.outputsCollapsibleButton)
        self.startButton.setObjectName("startButton")
        self.gridLayout.addWidget(self.startButton, 2, 0, 1, 2)
        self.detailedAirwaysCheckBox = QtWidgets.QCheckBox(self.outputsCollapsibleButton)
        self.detailedAirwaysCheckBox.setEnabled(False)
        self.detailedAirwaysCheckBox.setObjectName("detailedAirwaysCheckBox")
        self.gridLayout.addWidget(self.detailedAirwaysCheckBox, 1, 0, 1, 1)
        self.verticalLayout.addWidget(self.outputsCollapsibleButton)
        self.advancedCollapsibleButton = ctkCollapsibleButton(LungCTSegmenter)
        self.advancedCollapsibleButton.setCollapsed(False)
        self.advancedCollapsibleButton.setObjectName("advancedCollapsibleButton")
        self.formLayout_3 = QtWidgets.QFormLayout(self.advancedCollapsibleButton)
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_3 = QtWidgets.QLabel(self.advancedCollapsibleButton)
        self.label_3.setObjectName("label_3")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.ThresholdRangeWidget = ctkRangeWidget(self.advancedCollapsibleButton)
        self.ThresholdRangeWidget.setMinimum(-1500.0)
        self.ThresholdRangeWidget.setMaximum(1000.0)
        self.ThresholdRangeWidget.setMinimumValue(-1000.0)
        self.ThresholdRangeWidget.setMaximumValue(-200.0)
        self.ThresholdRangeWidget.setObjectName("ThresholdRangeWidget")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.ThresholdRangeWidget)
        self.verticalLayout.addWidget(self.advancedCollapsibleButton)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)

        self.retranslateUi(LungCTSegmenter)
        LungCTSegmenter.mrmlSceneChanged['vtkMRMLScene*'].connect(self.inputVolumeSelector.setMRMLScene)
        LungCTSegmenter.mrmlSceneChanged['vtkMRMLScene*'].connect(self.rightLungPlaceWidget.setMRMLScene)
        LungCTSegmenter.mrmlSceneChanged['vtkMRMLScene*'].connect(self.outputSegmentationSelector.setMRMLScene)
        LungCTSegmenter.mrmlSceneChanged['vtkMRMLScene*'].connect(self.leftLungPlaceWidget.setMRMLScene)
        LungCTSegmenter.mrmlSceneChanged['vtkMRMLScene*'].connect(self.tracheaPlaceWidget.setMRMLScene)
        QtCore.QMetaObject.connectSlotsByName(LungCTSegmenter)

    def retranslateUi(self, LungCTSegmenter):
        _translate = QtCore.QCoreApplication.translate
        self.inputsCollapsibleButton.setText(_translate("LungCTSegmenter", "Inputs"))
        self.label.setText(_translate("LungCTSegmenter", "Input volume:"))
        self.inputVolumeSelector.setToolTip(_translate("LungCTSegmenter", "Pick the input (CT Lung) to the algorithm."))
        self.label_2.setText(_translate("LungCTSegmenter", "Output segmentation:"))
        self.outputSegmentationSelector.setToolTip(_translate("LungCTSegmenter", "Pick the output segmentatioon or create a new one."))
        self.outputSegmentationSelector.setBaseName(_translate("LungCTSegmenter", "Lung segmentation"))
        self.outputSegmentationSelector.setNoneDisplay(_translate("LungCTSegmenter", "Create new segmentation"))
        self.outputsCollapsibleButton.setText(_translate("LungCTSegmenter", "Segmentation"))
        self.applyButton.setToolTip(_translate("LungCTSegmenter", "Run the algorithm."))
        self.applyButton.setText(_translate("LungCTSegmenter", "Apply"))
        self.cancelButton.setToolTip(_translate("LungCTSegmenter", "Cancel the current segmentation process."))
        self.cancelButton.setText(_translate("LungCTSegmenter", "Cancel"))
        self.instructionsLabel.setToolTip(_translate("LungCTSegmenter", "Find instructzions here during the segmentation procedure.  "))
        self.adjustPointsGroupBox.setTitle(_translate("LungCTSegmenter", "Adjust points "))
        self.label_4.setToolTip(_translate("LungCTSegmenter", "Place a marker somewhere on the right lung. "))
        self.label_4.setText(_translate("LungCTSegmenter", "Right lung:"))
        self.rightLungPlaceWidget.setToolTip(_translate("LungCTSegmenter", "Add additional or adjust existing right lung markers. "))
        self.label_5.setStatusTip(_translate("LungCTSegmenter", "Place a marker somewhere on the left lung. "))
        self.label_5.setText(_translate("LungCTSegmenter", "Left lung:"))
        self.leftLungPlaceWidget.setToolTip(_translate("LungCTSegmenter", "Add additional or adjust existing left lung markers. "))
        self.leftLungPlaceWidget.setStatusTip(_translate("LungCTSegmenter", "Place a marker somewhere on the left lung. "))
        self.label_6.setToolTip(_translate("LungCTSegmenter", "Place a marker on the upper trachea (above upper thoracic aperture) "))
        self.label_6.setText(_translate("LungCTSegmenter", "Other:"))
        self.tracheaPlaceWidget.setToolTip(_translate("LungCTSegmenter", "Add additional or adjust existing trachea markers. "))
        self.toggleSegmentationVisibilityButton.setToolTip(_translate("LungCTSegmenter", "Press this button to show and unshow the visibility of segments in 2D view"))
        self.toggleSegmentationVisibilityButton.setText(_translate("LungCTSegmenter", "Toggle segments visibility"))
        self.startButton.setToolTip(_translate("LungCTSegmenter", "Press this button to start the segmentation process. "))
        self.startButton.setText(_translate("LungCTSegmenter", "Start"))
        self.detailedAirwaysCheckBox.setText(_translate("LungCTSegmenter", "Produce detailed airways"))
        self.advancedCollapsibleButton.setText(_translate("LungCTSegmenter", "Advanced"))
        self.label_3.setText(_translate("LungCTSegmenter", "Lung intensity range:"))
        self.ThresholdRangeWidget.setToolTip(_translate("LungCTSegmenter", "Select the threshold range to identify lung parenchyma.  In doubt do not touch. "))
from ctkCollapsibleButton import ctkCollapsibleButton
from ctkCollapsibleGroupBox import ctkCollapsibleGroupBox
from ctkFittedTextBrowser import ctkFittedTextBrowser
from ctkRangeWidget import ctkRangeWidget
from qMRMLNodeComboBox import qMRMLNodeComboBox
from qMRMLWidget import qMRMLWidget
from qSlicerMarkupsPlaceWidget import qSlicerMarkupsPlaceWidget


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    LungCTSegmenter = QtWidgets.qMRMLWidget()
    ui = Ui_LungCTSegmenter()
    ui.setupUi(LungCTSegmenter)
    LungCTSegmenter.show()
    sys.exit(app.exec_())
