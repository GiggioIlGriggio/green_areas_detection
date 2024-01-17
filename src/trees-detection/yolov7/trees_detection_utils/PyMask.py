""" 
Code for converting shp files into binary masks (i.e. gray-scale png images).

For seeing the usage, have a look to `shp_into_binary.ipynb`.
"""

import sys, os

import shapefile

from PIL import Image, ImageDraw

# per ogni immagine costruisce unfile ascii con le coord pixel degli elementi della categoria
def shpToTree(imgName, shpName):
    try:
        tf = tfw(imgName)
    except Exception as err:
        print(err.strerror)
        return

    try:
        shape = shapefile.Reader(shpName)
    except Exception as err:
        print(err.args[0])
        return
    if shape.dbf is None or shape.shp is None:
        print('Errore apertura file ' + shpName)
        return

    outFile = os.path.splitext(imgName)[0] +'_tree.txt'
    f = open(outFile, 'w')
    if f is None:
        print(' impossibile aprire: ' + outFile)
        return

    cnt = 0
    for i in range(0, shape.numShapes):
        feature = shape.shapeRecords()[i]
        mes = 'Feature ' + str(i) + '\n'

        x = 0.5 * ( feature.shape.bbox[0] + feature.shape.bbox[2])
        y = 0.5 * ( feature.shape.bbox[1] + feature.shape.bbox[3])
        w = tf.ScaletoImg( feature.shape.bbox[2] - feature.shape.bbox[0])
        h = tf.ScaletoImg(feature.shape.bbox[3] - feature.shape.bbox[1])
        xi, yi = tf.toImg(x, y)
        mes = "{:.1f}".format(xi) + ', ' + "{:.1f}".format(yi) + ', ' + \
              "{:.1f}".format(w) +  ', ' + "{:.1f}".format(h)+'\n'

        for p in feature.shape.points:
            xi, yi = tf.toImg(p[0], p[1])
            mes = mes + "{:.1f}".format(xi) + ' ' + "{:.1f}".format(yi) + '\n'
        cnt += 1
        print(mes, file=f)
    print(outFile + ' ' + str(cnt) + ' entità')

class tfw:
    def __init__(self, nome):
        #print('CIAO')
        img = Image.open(nome)
        #print('EHI')

        self.width = img.width
        self.height = img.height
        self.nome = os.path.splitext(nome)[0] + '.tfw'

        #print('CIAO')
        with open(self.nome, 'r') as _tfw:
            line = _tfw.readlines()
            self.sx = float(line[0])
            self.sy = float(line[3])
            self.tx = float(line[4])
            self.ty = float(line[5])
        #print('EHI')

    def toImg(self, x, y):
        xi = (x -self.tx) / self.sx
        yi = (y -self.ty) / self.sy
        return xi, yi

    def ScaletoImg(self, d):
        return d / self.sx

# per ogni immagine costruisce una maschera con in bianco i pixel della categoria
def shpToMask(imgName, shpName, out_folder=''):
    try:
        tf = tfw(imgName)
    except Exception as err:
        print(err.strerror + ' ' + imgName)
        return

    try:
        shape = shapefile.Reader(shpName)
    except Exception as err:
        print(err.args[0])
        return
    if shape.dbf is None or shape.shp is None:
        print('Errore apertura file ' + shpName)
        return

    outFile = out_folder + os.path.splitext(imgName)[0] +'_mask.png'

    img = Image.new('RGB', (tf.width, tf.height), "black")
    drw = ImageDraw.Draw(img)

    cnt = 0
    for f in range(0, shape.numShapes):
        xy = []
        feature = shape.shapeRecords()[f]
        for p in feature.shape.points:
            xi, yi = tf.toImg(p[0], p[1])
            xy.append(xi)
            xy.append(yi)
        if len(xy) > 0:
            cnt += 1
            drw.polygon(xy, fill="white", outline="black")
    print(outFile + ' ' + str(cnt) + ' entità')
    img.save(outFile, quality=95)

def shpToPoly(imgName, shpName):
    try:
        tf = tfw(imgName)
    except Exception as err:
        print('Errore ' + err.strerror)
        return

    try:
        shape = shapefile.Reader(shpName)
    except Exception as err:
        print(err.args[0])
        return
    if shape.dbf is None or shape.shp is None:
        print('Errore apertura file ' + shpName)
        return

    outFile = os.path.splitext(imgName)[0] +'_mask.txt'
    f = open(outFile, 'w')
    if f is None:
        print(' impossibile aprire: ' + outFile)
        return

    cnt = 0
    for i in range(0, shape.numShapes):
        feature = shape.shapeRecords()[i]
        mes = 'Feature ' + str(i) + ' ' + str(len(feature.shape.points)) + '\n'
        for p in feature.shape.points:
            xi, yi = tf.toImg(p[0], p[1])
            mes = mes + "{:.1f}".format(xi) + ' ' + "{:.1f}".format(yi) + '\n'
        cnt += 1
        print(mes, file=f)
    print(outFile + ' ' + str(cnt) + ' entità')

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Uso pymask.py cmd img_path dis_path')
        print('\tcmd: 0: bitmap mask, 1: polygon, 2: tree')
        print('\timg_path: nome completo del file tif')
        print('\tdis_path: path del disegno, senza estensione')
        exit(0)

    cmd = int(sys.argv[1])
    imgName = sys.argv[2]
    disName = sys.argv[3]
    if cmd == 0:
        shpToMask(imgName, disName)
    elif cmd == 1:
        shpToPoly(imgName, disName)
    else:
        shpToTree(imgName, disName)


