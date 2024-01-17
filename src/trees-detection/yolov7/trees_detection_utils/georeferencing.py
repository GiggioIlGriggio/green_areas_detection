import shapefile
import os
import sys
from geojson import Point, Polygon, Feature, FeatureCollection, dump



################### UTILITIES
crs = {
    "type": "name",
    "properties": {
        "name": "urn:ogc:def:crs:EPSG::32632"
    }
}

class _tfw:
    def __init__(self, nome):

        self.nome = os.path.splitext(nome)[0] + '.tfw'

        with open(self.nome, 'r') as _tfw:
            line = _tfw.readlines()
            self.sx = float(line[0])
            self.sy = float(line[3])
            self.tx = float(line[4])
            self.ty = float(line[5])

    def fromImg(self, xi, yi):
        x = xi * self.sx + self.tx
        y = yi * self.sy + self.ty
        return [x, y]

    def toImg(self, x, y):
        xi = (x -self.tx) / self.sx
        yi = (y -self.ty) / self.sy
        return xi, yi

    def ScaletoImg(self, d):
        return d / self.sx

class _shape:
    def __init__(self, nome, type):
        self.nome = nome
        self.shp = None
        if type == 'polygon':
            sType = shapefile.POLYGON
        elif type == 'point':
            sType = shapefile.POINT
        else:
            raise Exception("Invalid geometry type")
        try:
            self.shp = shapefile.Writer(nome, sType)
        except:
            err = True

    def addField(self, nome, type, dec=0):
        self.shp.field(nome, type, decimal=dec)

    def addPoly(self, coord, fields):
        self.shp.poly(coord)
        self.shp.record(*fields)

    def addPoint(self, coord, fields):
        self.shp.point(coord[0][0], coord[0][1])
        self.shp.record(*fields)

    def close(self):
        self.shp.close()


def _decodeCoord(pts, tf):
    coord = []
    for pt in pts:
        ptf = tf.fromImg(float(pt[0]), float(pt[1]))
        coord.append(ptf)
    return coord

def _createShapePoint(nomeImg, data, conf, tf):
    # open shape file
    nomeShape = os.path.basename(nomeImg).split('.')[0] + '_treePoints.shp'
    #nomeShape = nomeJson.split('.')[0] + '.shp'
    try:
        shpw = _shape(nomeShape, 'point')
    except:
        print('impossibile aprire il file shape')
        return
    shpw.addField('tavola', 'C')
    shpw.addField('precision', 'N', 3)

    nomeTavola = os.path.basename(nomeImg).split('.')[0]

    for i in range(0, len(data)):
        coord = _decodeCoord(data[i], tf)
        fields = []
        fields.append(nomeTavola)
        fields.append(conf[i])
        shpw.addPoint(coord, fields)

    shpw.close()

def _createShapePolygon(nomeImg, data, conf, tf):
    # open shape file
    nomeShape = os.path.basename(nomeImg).split('.')[0] + '_treeBox.shp'
    try:
        shpw = _shape(nomeShape, 'polygon')
    except:
        print('impossibile aprire il file shape')
        return

    shpw.addField('tavola', 'C')
    shpw.addField('precision', 'N', 3)

    nomeTavola = os.path.basename(nomeImg).split('.')[0]

    for i in range(0, len(data)):
        coord = _decodeCoord(data[i], tf)
        fields = []
        fields.append(nomeTavola)
        fields.append(conf[i])
        shpw.addPoly(coord, fields)

    # Closing file
    shpw.close()

def _createGeoJsonPoint(nomeImg, data, conf, tf, points_geojson_output_path):
    nomeTavola = os.path.basename(nomeImg).split('.')[0]
    features = []
    for i in range(0, len(data)):
        p = _decodeCoord(data[i], tf)
        point = Point((p[0][0], p[0][1]))

        features.append(Feature(geometry=point, properties={"tavola": nomeTavola, 'precision': str(conf[i])}))

    feature_collection = FeatureCollection(features, crs=crs)

    nomeGeoJson = points_geojson_output_path
    with open(nomeGeoJson, 'w') as f:
        dump(feature_collection, f)

def _createGeoJsonPolygon(nomeImg, data, conf, tf, boxes_geojson_output_path):
    nomeTavola = os.path.basename(nomeImg).split('.')[0]
    features = []

    for i in range(0, len(data)):
        coords = _decodeCoord(data[i], tf)
        cc = []
        for c in coords:
            cc.append((c[0], c[1]))

        poly = Polygon([cc])
        features.append(Feature(geometry=poly, properties={"tavola": nomeTavola, 'confidence': str(conf[i])}))

    feature_collection = FeatureCollection(features, crs=crs)
    nomeGeoJson = boxes_geojson_output_path #os.path.basename(nomeImg).split('.')[0] + '_treesBoxes.geojson'
    with open(nomeGeoJson, 'w') as f:
        dump(feature_collection, f)

def _runTreePoint(nomeTxt, nomeImg, points_geojson_output_path):
    #read tfw file
    try:
        tf = _tfw(nomeImg)
    except:
        print('impossibile aprire il file tfw')
        return

    # read txt file
    data = []
    conf = []
    try:
        with open(nomeTxt, 'r') as f:
            for l in f:
                s = l.split(' ')
                data.append([[float(s[0]), float(s[1])]])
                conf.append(float(s[4]))
    except:
        print('impossibile aprire il file txt')
        return

    #_createShapePoint(nomeImg, data, conf, tf)

    _createGeoJsonPoint(nomeImg, data, conf, tf, points_geojson_output_path)

    # Closing file
    f.close()

def _runTreeBound(nomeTxt, nomeImg, boxes_geojson_output_path):
    #read tfw file
    try:
        tf = _tfw(nomeImg)
    except:
        print('impossibile aprire il file tfw')
        return

    # read json file
    data = []
    conf = []
    try:
        with open(nomeTxt, 'r') as f:
            for l in f:
                s = l.split(' ')
                px = float(s[0])
                py = float(s[1])
                dx = float(s[2]) / 2
                dy = float(s[3]) / 2
                data.append([[px-dx, py-dy], [px+dx, py-dy], [px+dx, py+dy], [px-dx, py+dy], [px-dx, py-dy]])
                conf.append(float(s[4]))
    except:
        print('impossibile aprire il file txt')
        return

    #_createShapePolygon(nomeImg, data, conf, tf)
    _createGeoJsonPolygon(nomeImg, data, conf, tf, boxes_geojson_output_path)

    # Closing file
    f.close()




################### GEOREFERENCING
def georeference(bbs_txt_path, tfw_file_path, points_geojson_output_path, boxes_geojson_output_path):
    """Convert the bounding boxes from pixel coordinates to geographical coordinates. This process of
    attaching geographical coordinates is called 'georeferencing'.

    It takes in input a `.txt` file containing trees bounding boxes, in pixel coordinates.  Each line 
    represents a bounding box, with structure 
               `'{x_center} {y_center} {w} {h} {confidence}\n'`

    It returns in output two `.geojson` files. 
        - One containing the georeferenced baricenter (i.e. the center) of each bounding box.
        - One containing the georeferenced boxes. It contains also the confidence scores among the 
          properties.

    This code was given to us by our GEOIN colleagues.

    Parameters
    ----------
    bbs_txt_path : str
        Path to the `.txt` file containing the bounding boxes.
    tfw_file_path : str
        Path to the image `.tfw` file.
    points_geojson_output_path : str
        Path to the geojson points output file.
    boxes_geojson_output_path : str
        Path to the geojson boxes output file.
    """

    # Create the georeferenced points file
    _runTreePoint(bbs_txt_path, tfw_file_path, points_geojson_output_path)

    # Create the georeferenced boxes file
    _runTreeBound(bbs_txt_path, tfw_file_path, boxes_geojson_output_path)
