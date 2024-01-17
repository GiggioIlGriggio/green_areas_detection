import json
import geojson
import shapely
import sys
import shapely
import sys



################### UTILITIES

class _Tfw:
  def __init__(self, filename):
    fTfw = open(filename, "r")
    tfw_raw_data = fTfw.read()
    self.deltaX = float(tfw_raw_data.split("\n")[0])
    self.deltaY = float(tfw_raw_data.split("\n")[3])
    self.x1 = float(tfw_raw_data.split("\n")[4])
    self.y1 = float(tfw_raw_data.split("\n")[5])
    fTfw.close()
  def pixelToGeo(self, x, y):
    xGeo = x * self.deltaX + self.x1
    yGeo = y * self.deltaY + self.y1
    return (xGeo, yGeo)

def _get_simple_polygons(geom):
  polygons = []
  if shapely.get_type_id(geom)==7: #GeometryCollection
    # print("GeometryCollection size ", len(geom.geoms))
    for g in geom.geoms:
      gPolygons = _get_simple_polygons(g)
      for gp in gPolygons:
        polygons.append(gp)
  elif shapely.get_type_id(geom)==6: #Multipolygon
    # print("Multipolygon size ", len(geom.geoms))
    for p in geom.geoms:
      polygons.append(p)
  elif shapely.get_type_id(geom)==3: #Polygon
    for i in geom.interiors:
      polygons.append(shapely.Polygon(i))
    polygons.append(shapely.Polygon(geom.exterior))
  return polygons

class _PolygonStruct:
    def __init__(self, polygon):
        self.polygon = polygon #base polygon, with only external ring
        self.holes = [] #list of internal rings
    def addHole(self, polygon):
        self.holes.append(polygon)
    def getCompletePolygon(self):
        #costruisce il poligono completo, externalRing+holes
        fixPolygon = self.polygon
        if shapely.is_ccw(self.polygon.exterior):
            fixPolygon = self.polygon.reverse()
        fixHoles = []
        for h in self.holes:
            if shapely.is_ccw(h.exterior):
                fixHoles.append(h.exterior)
            else:
                revHole = h.reverse()
                fixHoles.append(revHole.exterior)
        return shapely.Polygon(fixPolygon.exterior, fixHoles)
    def getExtRingWkt(self):
        return self.polygon.wkt
    def getHolesWkt(self):
        result = ""
        for h in self.holes:
            result += h.wkt
            result += " "
        return result
    def isValid(self):
        return shapely.is_valid(self.polygon)
    def __str__(self):
        txt = str(self.polygon) + " " + str(self.holes)
        return txt

class _Hole:
    def __init__(self, polygon, externalRings):
        self.polygon = polygon
        self.externalRings = externalRings
        self.assigned = False
    def minAreaExternalRings(self):
        minArea = sys.float_info.max
        minAreaEr = None
        for er in self.externalRings:
            if shapely.area(er) < minArea:
                minAreaEr = er
                minArea = shapely.area(er)
        return minAreaEr
    def __str__(self):
        txt = str(len(self.externalRings)) + " " + str(self.assigned) + " " + str(self.polygon)
        return txt
   



################### GEOREFERENCING 

def georeference(polylines, tfw_file_path, output_file_path):
    """Convert poly-lines from pixel coordinates to geographical coordinates.

    Basically, it takes in input a `.json` file and it produces a `.geojson` file. This process of
    attaching geographical coordinates is called 'georeferencing'.

    This code was given to us by our GEOIN colleagues.

    Parameters
    ----------
    polylines : list
        Poly-lines
    tfw_file_path : str
        Path to the `.twf` image file.
    output_file_path : _type_
        Path to the output `.geojson` file.
    """

    # Opening JSON file
    #polylines = open(polylines_json_path)
    # returns JSON object as 
    # a dictionary
    #polylines = json.load(polylines)

    # Parsing TFW
    tfw = _Tfw(tfw_file_path)

    # Iterating through the json list
    #print("Num.Polylines: ", len(polylines))
    polygons = []
    countValid = 0
    for polyline in polylines:
        geoPolyline = []
        for point in polyline:
            geoPolyline.append(tfw.pixelToGeo(point[0], point[1]))
        polygon = shapely.Polygon(geoPolyline)
        if shapely.is_valid(polygon):  # polyline is a simple and valid polygon
            countValid += 1
            polygons.append(polygon)
        else:  # polyline is not a valid polygon
            valid_geom = shapely.make_valid(polygon)
            simple_valid_geom = _get_simple_polygons(valid_geom)
            for p in simple_valid_geom:
                polygons.append(p)

    # test validity on simple polygons
    count_simple = 0
    count_valid = 0
    for p in polygons:
        if shapely.is_valid(p):
            count_valid += 1
        if len(p.interiors) == 0 and shapely.get_type_id(p) == 3:
            count_simple += 1
    #print(len(polygons), count_simple, count_valid)

    # combine polygon-holes
    nPolygons = len(polygons)
    completePolygons = []
    holes = []
    for i in range(nPolygons):
        outerPolygons = []
        pI = polygons[i]
        for j in range(nPolygons):
            if i != j:
                pJ = polygons[j]
                if shapely.within(pI, pJ):
                    outerPolygons.append(pJ)
        if (len(outerPolygons) % 2) == 0:
            completePolygons.append(_PolygonStruct(pI))
        else:
            h = _Hole(pI, outerPolygons)
            holes.append(h)
    for i in range(len(completePolygons)):
        for j in range(len(holes)):
            h = holes[j]
            if h.assigned:
                continue
            er = h.minAreaExternalRings()
            if er != None and shapely.equals(er, completePolygons[i].polygon):
                completePolygons[i].addHole(h.polygon)
                h.assigned = True
    # check all assigned
    countPolygonWithHoles = 0
    countHoles = 0
    for p in completePolygons:
        if len(p.holes) > 0:
            countHoles += len(p.holes)
            countPolygonWithHoles += 1
    #print(len(holes), countHoles, countPolygonWithHoles)

    # save polygons in geojson
    features = []
    for pStruct in completePolygons:
        p = pStruct.getCompletePolygon()
        f = shapely.to_geojson(p)
        features.append(geojson.Feature(geometry=geojson.loads(f)))
    feature_collection = geojson.FeatureCollection(features)
    text_geojson = '{"type":"FeatureCollection","features":' + geojson.dumps(features) + "}"

    with open(output_file_path, "w") as f:
        f.write(text_geojson)

