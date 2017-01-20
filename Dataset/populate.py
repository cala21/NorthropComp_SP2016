from PIL import Image
from os import listdir
import time
import datetime
import base64
import MySQLdb


Labeled = sorted(listdir("./FinalLabeledData"))
Raw = sorted(listdir("./FinalRawData"))
INSERT_INTO = "INSERT INTO goes_data (Id, PixelData, date, PixelLabels, NextFrame, PrevFrame) VALUES (%s,%s,%s,%s,%s,%s)"
#INSERT_INTO = "INSERT INTO goes_data (Id, PixelData, PixelLabels, date, NextFrame, PrevFrame) VALUES (%s %s %s %s %s %s)"


id = 1
prev = ""
prevd = {}

sql = "use goes;\n"
datePattern  = "%H%M%Y%m%d"
data = []

db = MySQLdb.connect("localhost", "root", "", "goes")
cursor = db.cursor()
cursor.execute("SELECT * FROM goes_data");

for rawFP, labelFP in zip(Raw,Labeled):
    #raw = Image.open("./FinalRawData/" + rawFP)
    #label = Image.open("./FinalLabeledData/" + labelFP)

    #p1 = raw.load()
    #p2 = label.load()
   

    d = rawFP[4:-12]


    try:
        #date  = datetime.datetime(d,datePattern)
        date = datetime.datetime.strptime( d, datePattern  ).strftime("%Y-%m-%d %H:%M:%S")
        #date = "FROM_UNIXTIME({})".format(str(int(int(time.mktime(time.strptime( d, datePattern  )))/1000)))
    except:
        continue

    p = prevd.get(prev)
    if not p:
        p = -1

    rawRaw = open("./FinalRawData/" + rawFP, "rb").read()
    rawLabel = open("./FinalLabeledData/" + labelFP, "rb").read()
    #rawRaw = MySQLdb.escape_string(str(base64.encodestring(open("./FinalRawData/" + rawFP, "rb").read())))
    #rawLabel = MySQLdb.escape_string(str(base64.encodestring(open("./FinalLabeledData/" + labelFP, "rb").read())))
    #vals = INSERT_INTO + "({},{},{},{},{},{})".format(id,rawRaw, rawLabel , date, None, p)


    #print(INSERT_INTO + str([id, rawRaw, rawLabel, date, -1, p]))
    cursor.execute(INSERT_INTO, (id,rawRaw,date,rawLabel, -1, p))
    #cursor.execute(INSERT_INTO, (id, rawRaw, rawLabel, date, -1, p))
    db.commit()

    #sql += vals

    prev = rawFP
    prevd[prev] = id
    id += 1



db.close()
