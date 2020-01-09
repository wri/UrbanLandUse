#add arterials

feature_name="Blocks"  # "Boundary", "Blocks" or "Roads" or "Areas"
group="Arterial" # "T0" or "T1_T3" or "Arterial"

# Set the directory where the input files are stored
directory='C:/Users/'+user+'/World Resources Institute/Urban Land Use - Documents/AUE Data and Maps/City Data/'+region+'/'+city+'/'+city+'_'+feature_name+'/'+city+'_'+group+'/'
#directory='C:/Users/'+user+'/World Resources Institute/Urban Land Use - Documents/AUE Data and Maps/WRI-NGS custom data from NYU/#WRI_Mexican_cities_submission'+'/'+city+'/'+city+'_'+feature_name+'/'+city+'_'+group+'/'

file = ''+city+'_Master_AR_Medians.shp'

# create vector layer object
vlayer = QgsVectorLayer(directory +file , file , "ogr")
print(file)

# add the layer to the registry
QgsMapLayerRegistry.instance().addMapLayer(vlayer, False)

# add the layer to the group
QgsProject.instance().layerTreeRoot().addLayer(vlayer)

#add study area

feature_name="Areas" 

# Set the directory where the input files are stored
directory='C:/Users/'+user+'/World Resources Institute/Urban Land Use - Documents/AUE Data and Maps/City Data/'+region+'/'+city+'/'+city+'_'+feature_name+'/'
#directory='C:/Users/'+user+'/World Resources Institute/Urban Land Use - Documents/AUE Data and Maps/WRI-NGS custom data from NYU/#WRI_Mexican_cities_submission'+'/'+city+'/'+city+'_'+feature_name+'/'+city+'_'+group+'/'

file = ''+city+'_studyArea.shp'

# create vector layer object
vlayer = QgsVectorLayer(directory +file , file , "ogr")
print(file)

# add the layer to the registry
QgsMapLayerRegistry.instance().addMapLayer(vlayer, False)

# add the layer to the group
QgsProject.instance().layerTreeRoot().addLayer(vlayer)

feature_name="Boundary"  # "Boundary", "Blocks" or "Roads" or "Areas"

if feature_name=='Boundary':
    feature_num='0' 
elif feature_name=='Blocks':
   feature_num='1'
print(feature_num)


group="T0" # "T0" or "T1_T3" or "Arterial"

# Set the directory where the input files are stored
directory='C:/Users/'+user+'/World Resources Institute/Urban Land Use - Documents/AUE Data and Maps/City Data/'+region+'/'+city+'/'+city+'_Blocks/'+city+'_'+group+'/'
#directory='C:/Users/'+user+'/World Resources Institute/Urban Land Use - Documents/AUE Data and Maps/WRI-NGS custom data from NYU/#WRI_Mexican_cities_submission'+'/'+city+'/'+city+'_Blocks'+'/'+city+'_'+group+'/'

# Add group for layers
iface.legendInterface().addGroup(''+city+'_'+feature_name+'_'+group+'')

mygroup = root.findGroup(''+city+'_'+feature_name+'_'+group+'')

# load vector layers
for files in os.listdir(directory):

    # load only the shapefiles
    if files.endswith(''+feature_num+'.shp'):

        # create vector layer object
        vlayer = QgsVectorLayer(directory + files, files , "ogr")
        print(files)

        # add the layer to the registry
        QgsMapLayerRegistry.instance().addMapLayer(vlayer, False)
        
        # add the layer to the group
        mygroup.addLayer(vlayer)

# Get the list of input files
fileList = os.listdir(directory)
 
# Copy the features from all the files in a new list
feats = []
feats_stored = []
for file in fileList:
    if file.endswith(''+feature_num+'.shp'):
        layer = QgsVectorLayer(directory + file, file, 'ogr')
        for feat in layer.getFeatures():
            geom = feat.geometry()
            attrs = feat.attributes()
            feature = QgsFeature()
            feature.setGeometry(geom)
            feature.setAttributes(attrs)
            feats.append(feature)
            feats_stored.append(feature)

# Get the Coordinate Reference System and the list of fields from the last input file
crs = layer.crs().toWkt()
field_list = layer.dataProvider().fields().toList()
 
# Create the merged layer by checking the geometry type of  the input files (for other types, please see the API documentation)
if layer.wkbType()==QGis.WKBPoint:
    v_layer = QgsVectorLayer('Point?crs=' + crs, "Boundary T0 Merged", "memory")
if layer.wkbType()==QGis.WKBLineString:
    v_layer = QgsVectorLayer('LineString?crs=' + crs, "Boundary T0 Merged", "memory")
if layer.wkbType()==QGis.WKBPolygon:
    v_layer = QgsVectorLayer('Polygon?crs=' + crs, "Boundary T0 Merged", "memory")

# Add the features to the merged layer
prov = v_layer.dataProvider()
prov.addAttributes(field_list)
v_layer.updateFields()
v_layer.startEditing()
prov.addFeatures(feats)
v_layer.commitChanges()
 
QgsMapLayerRegistry.instance().addMapLayer(v_layer)

group="T1_T3" # "T0" or "T1_T3" or "Arterial"

# Set the directory where the input files are stored
directory='C:/Users/'+user+'/World Resources Institute/Urban Land Use - Documents/AUE Data and Maps/City Data/'+region+'/'+city+'/'+city+'_Blocks/'+city+'_'+group+'/'
#directory='C:/Users/'+user+'/World Resources Institute/Urban Land Use - Documents/AUE Data and Maps/WRI-NGS custom data from NYU/#WRI_Mexican_cities_submission'+'/'+city+'/'+city+'_Blocks'+'/'+city+'_'+group+'/'

# Add group for layers
iface.legendInterface().addGroup(''+city+'_'+feature_name+'_'+group+'')

root = QgsProject.instance().layerTreeRoot()

mygroup = root.findGroup(''+city+'_'+feature_name+'_'+group+'')

# load vector layers
for files in os.listdir(directory):

    # load only the shapefiles
    if files.endswith(''+feature_num+'.shp'):

        # create vector layer object
        vlayer = QgsVectorLayer(directory + files, files , "ogr")
        print(files)

        # add the layer to the registry
        QgsMapLayerRegistry.instance().addMapLayer(vlayer, False)
        # fix for QGIS 3
        # QgsProject.instance().addMapLayer(vlayer, False)
        
        # add the layer to the group
        mygroup.addLayer(vlayer)

# Get the list of input files
fileList = os.listdir(directory)
 
# Copy the features from all the files in a new list
feats = []
for file in fileList:
    if file.endswith(''+feature_num+'.shp'):
        layer = QgsVectorLayer(directory + file, file, 'ogr')
        for feat in layer.getFeatures():
            geom = feat.geometry()
            attrs = feat.attributes()
            feature = QgsFeature()
            feature.setGeometry(geom)
            feature.setAttributes(attrs)
            feats.append(feature)
            feats_stored.append(feature)

# Get the Coordinate Reference System and the list of fields from the last input file
crs = layer.crs().toWkt()
field_list = layer.dataProvider().fields().toList()
 
# Create the merged layer by checking the geometry type of  the input files (for other types, please see the API documentation)
if layer.wkbType()==QGis.WKBPoint:
    v_layer = QgsVectorLayer('Point?crs=' + crs, "Boundary T1_T3 Merged", "memory")
if layer.wkbType()==QGis.WKBLineString:
    v_layer = QgsVectorLayer('LineString?crs=' + crs, "Boundary T1_T3 Merged", "memory")
if layer.wkbType()==QGis.WKBPolygon:
    v_layer = QgsVectorLayer('Polygon?crs=' + crs, "Boundary T1_T3 Merged", "memory")
 
# Add the features to the merged layer
prov = v_layer.dataProvider()
prov.addAttributes(field_list)
v_layer.updateFields()
v_layer.startEditing()
prov.addFeatures(feats)
v_layer.commitChanges()
 
QgsMapLayerRegistry.instance().addMapLayer(v_layer)

# Merge T0 and T1_T3 boundaries
if layer.wkbType()==QGis.WKBPoint:
    v_layer = QgsVectorLayer('Point?crs=' + crs, city+'_Locales_Merged', "memory")
if layer.wkbType()==QGis.WKBLineString:
    v_layer = QgsVectorLayer('LineString?crs=' + crs, city+'_Locales_Merged', "memory")
if layer.wkbType()==QGis.WKBPolygon:
    v_layer = QgsVectorLayer('Polygon?crs=' + crs, city+'_Locales_Merged', "memory")
    
# Add the features to the merged layer
prov = v_layer.dataProvider()
prov.addAttributes(field_list)
v_layer.updateFields()
v_layer.startEditing()
prov.addFeatures(feats_stored)
v_layer.commitChanges()

QgsMapLayerRegistry.instance().addMapLayer(v_layer)

# add centroid coordinates to Locales_Merged file
from PyQt4.QtCore import QVariant

# add centroid coordinates to Locales_Merged file
layerName = city+'_Locales_Merged'
layer = QgsMapLayerRegistry.instance().mapLayersByName(layerName)[0]

print layer.name()
fields = layer.pendingFields()
#for field in fields:
    #print field.displayName()

layer.startEditing()
layer.dataProvider().addAttributes([QgsField('lat', QVariant.Double),QgsField('long', QVariant.Double)])
layer.commitChanges()

index_lat = layer.dataProvider().fieldNameIndex('lat')
index_long = layer.dataProvider().fieldNameIndex('long')
print (index_lat, index_long)
assert index_lat != -1
assert index_long != -1

layer.startEditing()
for f in layer.getFeatures():
    cent_pt = f.geometry().centroid().asPoint()
    #print cent_pt.x(), cent_pt.y()
    layer.changeAttributeValue(f.id(), index_lat, cent_pt.y())
    layer.changeAttributeValue(f.id(), index_long, cent_pt.x())

layer.commitChanges()

