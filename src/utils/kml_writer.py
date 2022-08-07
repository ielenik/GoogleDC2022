header ='<?xml version="1.0" encoding="UTF-8"?>\n'\
        '<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">\n'\
        '<Folder>\n'\
        '    <name>%s</name>\n'\
        '    <open>1</open>\n'\
        '    <Document>\n'\
        '        <name>submission.kml</name>\n'\
        '        <Folder id="submission">\n'\
        '            <name>submission</name>\n'
footer ='		</Folder>\n'\
        '    </Document>\n'\
        '</Folder>\n'\
        '</kml>\n'

track = '<Placemark id="plot">\n'\
        '	<name>%s</name>\n'\
        '	<Style>\n'\
        '		<LineStyle>\n'\
        '			<color>%s</color>\n'\
        '			<width>1</width>\n'\
        '		</LineStyle>\n'\
        '	</Style>\n'\
        '	<LineString id="poly_plot">\n'\
        '		<tessellate>1</tessellate>\n'\
        '		<coordinates>\n'\
        '			%s\n'\
        '		</coordinates>\n'\
        '	</LineString>\n'\
        '</Placemark\n>'
pin = '<Placemark id="plot">\n'\
        '	<name>%s</name>\n'\
        '	<Point><coordinates>%s</coordinates></Point>\n'\
        '</Placemark\n>'

class KMLWriter:
    def __init__(self, path, name):
        self.sourceFile = open(path, 'w')
        print(header%name, file = self.sourceFile)
    def finish(self):
        print(footer, file = self.sourceFile)
        self.sourceFile.close()
    def addFolder(self,name):
        print('<Folder><open>0</open><name>',name,'</name>', file = self.sourceFile)
    def closeFolder(self):
        print('</Folder>', file = self.sourceFile)
    def addPoints(self,points):
        self.addFolder('Points')
        for i in range(0,len(points),1):
            p = points[i]
            print(pin%(str(i),(str(float(p[1]))+','+str(float(p[0])))), file = self.sourceFile)
        self.closeFolder()

    def addTrack(self,name,color,points, drawPoints = True):
        pts = ''
        for p in points:
            pts += str(float(p[1]))+','+str(float(p[0]))+',1 '
        pts = pts[:-1]
        print(track%(name,color,pts), file = self.sourceFile)
        if drawPoints:
            print('<Folder><open>0</open>', file = self.sourceFile)
            for i in range(0,len(points),10):
                p = points[i]
                print(pin%(str(i),(str(float(p[1]))+','+str(float(p[0])))), file = self.sourceFile)
            print('</Folder>', file = self.sourceFile)

