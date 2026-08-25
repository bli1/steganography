import os, math, time, hashlib, binascii
import cv2, numpy as np, pywt, xlwt
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes, serialization

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = HERE if os.path.isdir(os.path.join(HERE, "Original_image")) else os.path.dirname(HERE)
CLASSIC_DIR = os.path.join(BASE_DIR, "Encoded_image")
SPATIAL_DIR = os.path.join(BASE_DIR, "Spatial_encoded")
DWTSS_DIR = os.path.join(BASE_DIR, "Spread_encoded")
OUT_DIR = os.path.join(BASE_DIR, "All_methods_three_round_JPEG")
ATTACK_DIR = os.path.join(OUT_DIR, "attacked_images")
EXCEL_PATH = os.path.join(OUT_DIR, "all_methods_three_round_JPEG_results.xls")
PRIVATE_KEY = os.path.join(BASE_DIR, "private_key.pem")
PUBLIC_KEY = os.path.join(BASE_DIR, "public_key.pem")

WATERMARK_TEXT = "ASys Encryption"
JPEG_QUALITIES = [80, 60, 40]
RUNS = 10
REPORT_TOTAL = 1100
BER_THRESHOLD = 0.25
DCT_Q, DCT_BITS_PER_BLOCK, DWT_Q = 8.0, 4, 8.0
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

def load_keys():
    with open(PRIVATE_KEY, "rb") as f: private = serialization.load_pem_private_key(f.read(), password=None)
    with open(PUBLIC_KEY, "rb") as f: public = serialization.load_pem_public_key(f.read())
    if private.key_size != 2048 or public.key_size != 2048: raise ValueError("RSA-2048 keys are required")
    if private.public_key().public_numbers() != public.public_numbers(): raise ValueError("RSA key pair does not match")
    return private, public

def sign(private):
    return private.sign(WATERMARK_TEXT.encode(), padding.PKCS1v15(), hashes.SHA256())

def verify(public, signature):
    try:
        public.verify(signature, WATERMARK_TEXT.encode(), padding.PKCS1v15(), hashes.SHA256()); return True
    except Exception: return False

def bits_to_bytes(bits):
    bits = np.asarray(bits, dtype=np.uint8); return np.packbits(bits[:(len(bits)//8)*8]).tobytes()

def ber(received, expected):
    received = np.asarray(received, dtype=np.uint8)
    if len(received) < len(expected): received = np.pad(received, (0, len(expected)-len(received)))
    return float(np.mean(received[:len(expected)] != expected))

def jpeg_three_rounds(image):
    result = image
    for quality in JPEG_QUALITIES:
        ok, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok: raise IOError("JPEG encoding failed")
        result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if result is None: raise IOError("JPEG decoding failed")
    return result

def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
    return 100.0 if mse == 0 else 10*math.log10(255.0**2/mse)

def ssim(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64); c1, c2 = (2.55**2), (7.65**2); vals=[]
    for c in range(3):
        x,y=a[:,:,c],b[:,:,c]; mx=cv2.GaussianBlur(x,(11,11),1.5); my=cv2.GaussianBlur(y,(11,11),1.5)
        vx=cv2.GaussianBlur(x*x,(11,11),1.5)-mx*mx; vy=cv2.GaussianBlur(y*y,(11,11),1.5)-my*my
        vxy=cv2.GaussianBlur(x*y,(11,11),1.5)-mx*my
        vals.append(np.mean(((2*mx*my+c1)*(2*vxy+c2))/((mx*mx+my*my+c1)*(vx+vy+c2)+1e-12)))
    return float(np.mean(vals))

def extract_lsb(im, n):
    flat=im.reshape(-1,3); out=np.zeros(n,dtype=np.uint8); k=min(n,max(0,len(flat)-32)); out[:k]=flat[32:32+k,0]&1; return out

def extract_dct(im, n):
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY); out=[]; pos=[(4,4),(3,3),(2,2),(5,5)]
    for y in range(gray.shape[0]//8):
        for x in range(gray.shape[1]//8):
            d=cv2.dct(np.float32(gray[y*8:y*8+8,x*8:x*8+8]))
            for r,c in pos[:DCT_BITS_PER_BLOCK]:
                out.append(int(np.round(d[r,c]/DCT_Q))&1)
                if len(out)>=n: return np.asarray(out,dtype=np.uint8)
    return np.pad(np.asarray(out,dtype=np.uint8),(0,n-len(out)))

def extract_dwt(im, n):
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32); _,(lh,_,_)=pywt.dwt2(gray,"haar")
    q=np.round(lh.reshape(-1)[:n]/DWT_Q).astype(np.int64)&1; return np.pad(q,(0,max(0,n-len(q))))

def extract_ss(im, n, key, use_dwt):
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
    flat=(pywt.dwt2(gray,"haar")[0] if use_dwt else gray).reshape(-1); chunk=len(flat)//n
    seed=int.from_bytes(hashlib.sha256(key).digest()[:4],"big"); out=np.zeros(n,dtype=np.uint8)
    for i in range(n):
        seg=flat[i*chunk:min((i+1)*chunk,len(flat))]; pn=np.random.RandomState(seed+i).choice([-1,1],size=len(seg))
        out[i]=int(np.sum(seg*pn)>=0)
    return out

def jobs():
    result=[]
    if os.path.isdir(CLASSIC_DIR):
        for folder in sorted(os.listdir(CLASSIC_DIR)):
            for method,name in (("LSB","LSB.png"),("DCT","DCT.png"),("DWT","DWT.png")):
                p=os.path.join(CLASSIC_DIR,folder,name)
                if os.path.isfile(p): result.append((method,folder,p))
    for method,base,name in (("Spatial SS",SPATIAL_DIR,"spatial_spread.png"),("DWT-SS",DWTSS_DIR,"spread.png")):
        if os.path.isdir(base):
            for folder in sorted(os.listdir(base)):
                p=os.path.join(base,folder,name)
                if os.path.isfile(p): result.append((method,folder,p))
    return result

def decode(method, im, signature_bits, fingerprint, public):
    if method=="LSB": received=extract_lsb(im,len(signature_bits)); expected=signature_bits
    elif method=="DCT": received=extract_dct(im,len(signature_bits)); expected=signature_bits
    elif method=="DWT": received=extract_dwt(im,len(signature_bits)); expected=signature_bits
    elif method=="Spatial SS": received=extract_ss(im,len(fingerprint),b"SPATIAL_KEY",False); expected=fingerprint
    else: received=extract_ss(im,len(fingerprint),b"KEY",True); expected=fingerprint
    error=ber(received,expected); exact=bool(np.array_equal(received,expected))
    auth=verify(public,bits_to_bytes(received)) if method in ("LSB","DCT","DWT") else exact
    return error < BER_THRESHOLD, exact, auth, error

def write_excel(details, summaries):
    wb=xlwt.Workbook(); sh=wb.add_sheet("Per_Image")
    heads=["Run","Method","Image","Detected","Exact","Authenticated","BER","PSNR","SSIM","Status"]
    for c,h in enumerate(heads): sh.write(0,c,h)
    for r,row in enumerate(details,1):
        for c,v in enumerate(row): sh.write(r,c,v)
    ss=wb.add_sheet("Summary"); heads=["Method","Runs","Total","Detected Avg","Detected Std","Detection Rate","Success Avg","Success Std","Success Rate","Failure Avg","Failure Std","BER Mean","BER Std","PSNR Mean","SSIM Mean"]
    for c,h in enumerate(heads): ss.write(0,c,h)
    for r,row in enumerate(summaries,1):
        for c,v in enumerate(row): ss.write(r,c,v)
    wb.save(EXCEL_PATH)

def main():
    os.makedirs(ATTACK_DIR,exist_ok=True); private,public=load_keys(); signature=sign(private)
    signature_bits=np.unpackbits(np.frombuffer(signature,dtype=np.uint8)); fingerprint=np.unpackbits(np.frombuffer(hashlib.sha256(signature).digest()[:4],dtype=np.uint8))
    all_jobs=jobs(); methods=["LSB","DCT","DWT","Spatial SS","DWT-SS"]; details=[]; per_run={m:[] for m in methods}
    start=time.perf_counter()
    for run in range(1,RUNS+1):
        stats={m:{"d":0,"s":0,"ber":[],"p":[],"q":[]} for m in methods}
        for index,(method,image_id,path) in enumerate(all_jobs,1):
            im=cv2.imread(path,cv2.IMREAD_COLOR)
            if im is None: details.append((run,method,image_id,False,False,False,1.0,"","","READ FAILED")); continue
            try:
                attacked=jpeg_three_rounds(im); detected,exact,auth,error=decode(method,attacked,signature_bits,fingerprint,public)
                p,q=psnr(im,attacked),ssim(im,attacked); x=stats[method]; x["d"]+=detected; x["s"]+=auth; x["ber"].append(error); x["p"].append(p); x["q"].append(q)
                details.append((run,method,image_id,detected,exact,auth,error,p,q,"OK"))
                if run==1:
                    folder=os.path.join(ATTACK_DIR,method.replace(" ","_").replace("-","_"),image_id); os.makedirs(folder,exist_ok=True); cv2.imwrite(os.path.join(folder,"jpeg_80_60_40.jpg"),attacked)
            except Exception as e: details.append((run,method,image_id,False,False,False,1.0,"","",repr(e)))
        for m in methods: per_run[m].append(stats[m])
        print(f"Run {run}/{RUNS} completed")
    summaries=[]
    for m in methods:
        data=per_run[m]; detected=np.array([x["d"] for x in data],float); success=np.array([x["s"] for x in data],float); failure=REPORT_TOTAL-success
        summaries.append((m,RUNS,REPORT_TOTAL,float(detected.mean()),float(detected.std()),float(detected.mean()/REPORT_TOTAL),float(success.mean()),float(success.std()),float(success.mean()/REPORT_TOTAL),float(failure.mean()),float(failure.std()),float(np.mean([np.mean(x["ber"]) for x in data if x["ber"]])),float(np.std([np.mean(x["ber"]) for x in data if x["ber"]])),float(np.mean([np.mean(x["p"]) for x in data if x["p"]])),float(np.mean([np.mean(x["q"]) for x in data if x["q"]]))))
    write_excel(details,summaries)
    print("Completed in %.2f minutes"%((time.perf_counter()-start)/60)); print("Excel:",EXCEL_PATH); print("Images:",ATTACK_DIR)

if __name__=="__main__": main()
