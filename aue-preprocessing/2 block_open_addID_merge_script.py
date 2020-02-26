
feature_name="Blocks"  # "Boundary", "Blocks" or "Roads"

if feature_name=='Boundary':
    feature_num='0' 
elif feature_name=='Blocks':
    feature_num='1'
print(feature_num)

group="T0" # "T0" or "T1_T3"

# Set the directory where the input files are stored
directory='C:/Users/'+user+'/World Resources Institute/Urban Land Use - Documents/AUE Data and Maps/City Data/'+region+'/'+city+'/'+city+'_Blocks/'+city+'_'+group+'/'

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
        
        # add the layer to the group
        mygroup.addLayer(vlayer)

group="T1_T3" # "T0" or "T1_T3"

# Set the directory where the input files are stored
directory='C:/Users/'+user+'/World Resources Institute/Urban Land Use - Documents/AUE Data and Maps/City Data/'+region+'/'+city+'/'+city+'_Blocks/'+city+'_'+group+'/'

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
        
        # add the layer to the group
        mygroup.addLayer(vlayer)


# seam where joining two scripts
#
#

feature_name="Blocks"  # "Boundary", "Blocks" or "Roads"

if feature_name=='Boundary':
    feature_num='0' 
elif feature_name=='Blocks':
    feature_num='1'
print(feature_num)

valid = 1
success = 0

# test members of selected group for type and name
children = []
if valid:
    group = root.findGroup(''+city+'_'+feature_name+'_'+'T0'+'')
    children = group.children()
    for child in group.children():
        name = child.name()
        if not isinstance(child, QgsLayerTreeLayer):
            valid = 0
            print "Non-layer item in selected group: " + name
        elif len(name) != 19:
            valid = 0
            print "Layer name is not 15 characters: " + name
    group = root.findGroup(''+city+'_'+feature_name+'_'+'T1_T3'+'')
    children.extend(group.children())
    for child in group.children():
        name = child.name()
        if not isinstance(child, QgsLayerTreeLayer):
            valid = 0
            print "Non-layer item in selected group: " + name
        elif len(name) != 19:
            valid = 0
            print "Layer name is not 15 characters: " + name


#test contents of group members, looking at fields and values
if valid:
    for child in children:
        if isinstance(child, QgsVectorLayer):
            valid = 0
            print "Layer is not a VectorLayer: " + name
        else:
            layer = child.layer()
            fields = layer.pendingFields()
            field_names = [field.name() for field in fields]
            if len(fields) != 1:
                # valid = 0
                print "Layer does not have precisely one field (but script will continue): " + child.name()
            elif fields[0].name() != "Land_Use" and fields[0].name() != "Land_use":
                valid = 0
                print "Layer (" + child.name() + ") has field not named \"Land_Use\": " + fields[0].name()
            elif fields[0].typeName() != "String":
                valid = 0
                print "Layer's field \"Land_Use\" is not of type String: " + child.name()
            else:
                for feature in layer.getFeatures():
                    land_use_string = feature.attributes()[0]
                    land_use_value = int(land_use_string)
                    if len(land_use_string) != 1 or land_use_value < 0 or land_use_value > 5:
                        valid = 0
                        print "Layer (" + child.name() + ") contains illegal value for \"Land_Use\": " + land_use_string
                    
            
        
    # if group and contents are valid, now perform edits that are reason for this script
    if valid:
        for child in children:
            # first add field "ID_string" to all layers in group
            name = child.name().split('.')[0] # to omit file extension and keep only numeric identifier
            print name
            layer = child.layer()
            fields = layer.pendingFields()
            field_names = [field.name() for field in fields]
            if len(fields) != 1:
                # valid = 0
                print "Skipping layer that already contains multiple fields: " + child.name()
            else:
                newFieldName = "ID_string"
                success = layer.dataProvider().addAttributes( [QgsField(newFieldName, QVariant.String)] )
                if not success:
                    valid = 0
                    print "Was unable to add \"ID_string\" field to layer: " + name
                else:
                    # next set value of "ID_string" to name of layer
                    layer.updateFields()
                    fieldIndex = layer.dataProvider().fieldNameIndex(newFieldName)
                    attrFeatMap = {}
                    attrMap = { fieldIndex : name } 
                    for feature in layer.getFeatures():
                        attrFeatMap[ feature.id() ] = attrMap
                    layer.dataProvider().changeAttributeValues( attrFeatMap )
        if success: 
            print 'success'

if success:
    feats = []
    # Set the directory where the input files are stored
    group="T0" # "T0" or "T1_T3"
    directory='C:/Users/'+user+'/World Resources Institute/Urban Land Use - Documents/AUE Data and Maps/City Data/'+region+'/'+city+'/'+city+'_Blocks/'+city+'_'+group+'/'

    # Get the list of input files
    fileList = os.listdir(directory)
     
    # Copy the features from all the files in a new list
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

    group="T1_T3"
    directory='C:/Users/'+user+'/World Resources Institute/Urban Land Use - Documents/AUE Data and Maps/City Data/'+region+'/'+city+'/'+city+'_Blocks/'+city+'_'+group+'/'
    fileList = os.listdir(directory)
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
                
     
    # Get the Coordinate Reference System and the list of fields from the last input file
    crs = layer.crs().toWkt()
    field_list = layer.dataProvider().fields().toList()
     
    # Create the merged layer by checking the geometry type of  the input files (for other types, please see the API documentation)
    if layer.wkbType()==QGis.WKBPoint:
        v_layer = QgsVectorLayer('Point?crs=' + crs, city+'_Blocks_Merged', "memory")
    if layer.wkbType()==QGis.WKBLineString:
        v_layer = QgsVectorLayer('LineString?crs=' + crs, city+'_Blocks_Merged', "memory")
    if layer.wkbType()==QGis.WKBPolygon:
        v_layer = QgsVectorLayer('Polygon?crs=' + crs, city+'_Blocks_Merged', "memory")
     
    # Add the features to the merged layer
    prov = v_layer.dataProvider()
    prov.addAttributes(field_list)
    v_layer.updateFields()
    v_layer.startEditing()
    prov.addFeatures(feats)
    v_layer.commitChanges()
     
    QgsMapLayerRegistry.instance().addMapLayer(v_layer)

