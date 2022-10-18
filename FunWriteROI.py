import struct
from zipfile import ZipFile

types ={"polygon":0, "rect":1, "oval":2, "line":3, "freeline":4, "polyline":5, "noRoi":6, "freehand":7, "traced":8,
        "angle":9, "point":10}

#HEADER Offsets
VERSION_OFFSET = 4
TYPE_OFFSET = 6
TOP_OFFSET = 8
LEFT_OFFSET = 10
BOTTOM_OFFSET = 12
RIGHT_OFFSET = 14
N_COORDINATES_OFFSET = 16
X1_OFFSET = 18
Y1_OFFSET = 22
X2_OFFSET = 26
Y2_OFFSET = 30
XD_OFFSET = 18
YD_OFFSET = 22
WIDTHD_OFFSET =26
HEIGHTD_OFFSET =30
SIZE_OFFSET = 18
STROKE_WIDTH_OFFSET = 34
SHAPE_ROI_SIZE_OFFSET = 36
STROKE_COLOR_OFFSET = 40
FILL_COLOR_OFFSET = 44
SUBTYPE_OFFSET = 48
OPTIONS_OFFST =50
ARROW_STYLE_OFFSET = 52
FLOAT_PARAM_OFFSET =52
POINT_TYPE_OFFSET =52
ARROW_HEAD_SIZE_OFFSET = 53
ROUNDED_RECT_ARC_SIZE_OFFSET = 54
POSITION_OFFSET = 56
HEADER2_OFFSET = 60
COORDINATES_OFFSET =64

#Header2 offsets
C_POSITION_OFFSET = 4
Z_POISTION_OFFSET = 8
T_POSITION_OFFSET = 12
NAME_OFFSET = 16
NAME_LENGTH_OFFSET =20
OVERLAY_LABEL_COLOR_OFFSET = 24
OVERLAY_FONT_SIZE = 28
GROUP_OFFSET = 30
IMAGE_OPACITY_OFFSET = 31
IMAGE_SIZE_OFFSET = 32
FLOAT_STROKE_WIDTH = 36
ROI_PROPS_OFFSET = 40
ROI_PROPS_LENGTH_OFFSET = 44
COUNTERS_OFFSET = 48

class RectRoi():
    def __init__(self, x, y, width, height,name = '',c_position=0,z_position = 0 ,t_position = 0):
        self.left = x
        self.top = y
        self.right = x+width
        self.bottom = y+height

        self.type = 'rect'
        if name == '':
            self.name = str(c_position) + str(x)+ '-'+ str(y)   #rounded to integer
        else:
            self.name = name
        self.c_position = c_position
        self.z_position = z_position
        self.t_position = t_position

        #other default value
        #self.options=0

def GetRoiData(roi):
    def putShort(offset, v):
        offset = int(offset)
        data[offset:offset + 2] = struct.pack('>h', v)

    def putFloat(offset, v):
        offset = int(offset)
        data[offset:offset + 4] = struct.pack('>f', v)

    def putInt(offset, v):
        offset = int(offset)
        data[offset:offset + 4] = struct.pack('>i', v)

    def putHeader2(roi, hdr2Offset):
        putInt(HEADER2_OFFSET, hdr2Offset)
        putInt(HEADER2_OFFSET + C_POSITION_OFFSET, roi.c_position)
        putInt(HEADER2_OFFSET + Z_POISTION_OFFSET, roi.z_position)
        putInt(HEADER2_OFFSET + T_POSITION_OFFSET, roi.t_position)
        # if overlay label color.....
        # if font.........
        putName(hdr2Offset)
        # if.. put stroke width
        # if... put Roi property size
        # if... put countersize
        # if... put group

    def putName(hdr2Offset):
        offset = hdr2Offset + HEADER2_SIZE
        NameLength = len(roi.name)
        putInt(hdr2Offset + NAME_OFFSET, offset)  # I think offset is equal to hdr2offset+name_offset
        putInt(hdr2Offset + NAME_LENGTH_OFFSET, NameLength)
        for i in range(NameLength):
            putShort(offset + i * 2, ord(roi.name[i]))

    #----------------------------------------------------------------------
    HEADER_SIZE = int(64)
    HEADER2_SIZE = int(64) #what is it?
    VERSION = int(228)

    type = types[roi.type]
    options = 0
    roiNameSize = len(roi.name)*2
    #roiProps =
    #Get roiPropsSize.....
    roiPropsSize = 0
    #if roi is composite write another way. See RoiEncoder.java
    n=0
    floatSize =0
    #if roi is PolygonRoi .....it would change floatSize. skip for now
    countersSize = 0
    #if roi is Point modify the countersSize...

    dataSize = HEADER_SIZE + HEADER2_SIZE + n*4 + floatSize + roiNameSize + roiPropsSize + countersSize
    data = bytearray(dataSize)
    data[0]=73; data[1]=111; data[2]=117; data[3]=116; #"Iout"
    putShort(VERSION_OFFSET,VERSION)
    data[TYPE_OFFSET] = type
    putShort(TOP_OFFSET,roi.top)
    putShort(LEFT_OFFSET,roi.left)
    putShort(BOTTOM_OFFSET,roi.bottom)
    putShort(RIGHT_OFFSET,roi.right)
    #if subPixelReslution and type is rect or oval...... put XD, YD,WIDTHD,HEIGHTD,Optionss
    #if n>65535 and type is not point..........
    #if type is point and n>65535 put SIZE, N_Coordinates
    #put Position. I haven't knew the position means in the roi yet.
    #if tpye is Rectangle check the Arc size and put arcSize
    #if roi is type of line put x1,y1,x2,y2. And if roi is arrow put subtype arrow, options, arrow style and arrow head size.
    #if roi is type of Point roi, put point type and stroke width
    #if roi is type of Rotate Rectangle or Ellipse, put Subtype, x1,y1,x2,y2, float_param
    #if version> 218.......
    #if n equal to zero and roi is type of textRoi..........
    #if n equal to zero and roi is type of ImageROI.........
    hdr2Offset= HEADER_SIZE+n*4+floatSize    #for those are not textROI and ImageROI
    putHeader2(roi,hdr2Offset)
    #if n lager than 0,........
    #Save overlay options
    return data

def WriteRois(roiList,expPath,fileName):
    if len(roiList)==1:
        roi = roiList[0]
        with open(expPath+roi.name+'.roi','wb') as fout:
            fout.write(GetRoiData(roi))
            fout.close()
    else:
        with ZipFile(expPath+fileName+'.zip','w') as zout:
            for i,roi in enumerate(roiList):
                zout.writestr(roi.name+".roi", GetRoiData(roi))
            zout.close()