"""Render the fill-level measurement on three real frames."""
import json, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path("/Users/jasonli/Documents/GitHub/ZenblenImageTesting")
sys.path.insert(0, str(ROOT / "active_pipeline"))
from smoothie_cv.config import Config
from smoothie_cv.detection import detect_container

cfg = Config(); cfg.yolo_weights = ROOT / "active_pipeline/checkpoints/yolo_standard_seg.pt"
IM = ROOT / "training/labeling/data/images"
OUT = ROOT / "press_assets"
FONT = cv2.FONT_HERSHEY_SIMPLEX
d = json.load(open(sys.argv[1]))
MED, CUT = d["median"], d["cutoff"]
rows = {r["id"]: r for r in d["rows"]}

def label(img, text, y, scale=0.5, color=(250,250,250)):
    (w,h),_ = cv2.getTextSize(text, FONT, scale, 2)
    cv2.rectangle(img,(8,y-h-8),(8+w+14,y+9),(18,18,18),-1)
    cv2.putText(img,text,(15,y),FONT,scale,color,2,cv2.LINE_AA); return img

def tile(fid):
    r = rows[fid]
    img = cv2.imread(str(IM / f"{fid}.jpg"))
    m,_ = detect_container(img, cfg)
    over = img.copy(); over[m>0] = (255,170,40)
    img = cv2.addWeighted(over,0.16,img,0.84,0)
    ty,by,h = r["top_y"], r["bot_y"], r["h"]
    cv2.line(img,(0,ty),(480,ty),(60,60,255),2)              # liquid top
    cv2.line(img,(0,by),(480,by),(90,220,90),2)              # gasket seat
    cy = int(round(by - CUT))
    for x in range(0,480,16):                                 # cutoff, dashed
        cv2.line(img,(x,cy),(x+9,cy),(70,200,255),2)
    cv2.arrowedLine(img,(444,ty),(444,by),(255,255,255),2,tipLength=.03)
    cv2.arrowedLine(img,(444,by),(444,ty),(255,255,255),2,tipLength=.03)
    lowf = h < CUT
    col = (60,60,255) if lowf else (90,220,90)
    img = label(img, f"{h} px   {'LOW' if lowf else 'OK'}", 30, .62, col)
    d_pct = 100 * (1 - h / MED)
    word = "below" if d_pct >= 0 else "above"
    img = label(img, f"{abs(d_pct):.1f}% {word} median {MED:.0f}", 58, .46)
    return cv2.resize(img,(330,440))

picks = sys.argv[2].split(",")
grid = np.hstack([tile(f) for f in picks])
banner = np.full((58,grid.shape[1],3),18,np.uint8)
cv2.putText(banner,"Fill level - mask height against the fixed gasket seat",(16,25),FONT,.62,(250,250,250),2,cv2.LINE_AA)
cv2.putText(banner,f"red = liquid top   green = seat   dashed = {CUT:.0f} px flag cutoff",
            (16,46),FONT,.44,(150,155,145),1,cv2.LINE_AA)
cv2.imwrite(str(OUT/"ex_level.png"), np.vstack([banner,grid]))
for f in picks: print(f, rows[f]["h"], "LOW" if rows[f]["h"]<CUT else "OK")
