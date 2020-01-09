from qgis.core import *
import os

from PyQt4.QtCore import QVariant

city="johannesburg"
region="SubSaharanAfrica"
user="eric.pietraszkiewicz"

root = QgsProject.instance().layerTreeRoot()
