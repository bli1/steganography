from pathlib import Path
import base64, io, math, time
import numpy as np, xlwt
from PIL import Image
from trustmark import TrustMark
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

HERE=Path(__file__).resolve().parent
ROOT=HERE if (HERE/"Original_image").is_dir() else HERE.parent
INPUT_DIR=ROOT/"TrustMark_encoded"
OUT_DIR=ROOT/"TrustMark_three_round_JPEG"
ATTACK_DIR=OUT_DIR/"attacked_images"
EXCEL_PATH=OUT_DIR/"TrustMark_three_round_JPEG_results.xls"
PRIVATE_KEY=ROOT/"private_key.pem"; PUBLIC_KEY=ROOT/"public_key.pem"
WATERMARK_TEXT="ASys Encryption"; QUALITIES=[80,60,40]; RUNS=10; REPORT_TOTAL=1100
EXTS={".png",".jpg",".jpeg",".bmp",".webp",".tif",".tiff"}

def sign(private): return private.sign(WATERMARK_TEXT.encode(),padding.PKCS1v15(),hashes.SHA256())
def verify(public,sig):
    try: public.verify(sig,WATERMARK_TEXT.encode(),padding.PKCS1v15(),hashes.SHA256()); return True
    except Exception: return False
def payload(sig):
    d=hashes.Hash(hashes.SHA256()); d.update(sig); return base64.b32encode(d.finalize()[:5]).decode().rstrip("=")
def attack(im):
    out=im.convert("RGB")
    for q in QUALITIES:
        b=io.BytesIO(); out.save(b,"JPEG",quality=q); b.seek(0); out=Image.open(b).convert("RGB").copy()
    return out
def metrics(a,b):
    x=np.asarray(a,dtype=np.float64); y=np.asarray(b,dtype=np.float64); mse=np.mean((x-y)**2); p=100.0 if mse==0 else 10*math.log10(255**2/mse)
    # global SSIM, consistent and dependency-free
    mx,my=x.mean(),y.mean(); vx,vy=x.var(),y.var(); cov=np.mean((x-mx)*(y-my)); c1,c2=2.55**2,7.65**2
    s=((2*mx*my+c1)*(2*cov+c2))/((mx*mx+my*my+c1)*(vx+vy+c2)+1e-12); return float(p),float(s)
def payload_ber(decoded,expected):
    if decoded is None: return 1.0
    a=np.unpackbits(np.frombuffer(str(decoded).encode(),dtype=np.uint8)); b=np.unpackbits(np.frombuffer(expected.encode(),dtype=np.uint8))
    if len(a)<len(b): a=np.pad(a,(0,len(b)-len(a)))
    return float(np.mean(a[:len(b)]!=b))
def write_excel(details,summary):
    wb=xlwt.Workbook(); sh=wb.add_sheet("Per_Image"); heads=["Run","Image","Present","Decoded","Expected","Exact","Authenticated","BER","PSNR","SSIM","Schema","Error"]
    for c,h in enumerate(heads): sh.write(0,c,h)
    for r,row in enumerate(details,1):
        for c,v in enumerate(row): sh.write(r,c,v)
    ss=wb.add_sheet("Summary"); heads=["Runs","Total","Detected Avg","Detected Std","Detection Rate","Success Avg","Success Std","Success Rate","Failure Avg","Failure Std","BER Mean","BER Std","PSNR Mean","SSIM Mean","Expected Payload"]
    for c,h in enumerate(heads): ss.write(0,c,h)
    for c,v in enumerate(summary): ss.write(1,c,v)
    wb.save(str(EXCEL_PATH))
def main():
    OUT_DIR.mkdir(parents=True,exist_ok=True); ATTACK_DIR.mkdir(parents=True,exist_ok=True)
    with open(PRIVATE_KEY,"rb") as f: private=serialization.load_pem_private_key(f.read(),password=None)
    with open(PUBLIC_KEY,"rb") as f: public=serialization.load_pem_public_key(f.read())
    if private.key_size!=2048 or public.key_size!=2048: raise RuntimeError("RSA-2048 required")
    sig=sign(private)
    if not verify(public,sig): raise RuntimeError("RSA verification failed")
    expected=payload(sig); print("Expected payload:",expected); model=TrustMark()
    images=sorted([p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in EXTS],key=lambda p:p.name.lower())
    details=[]; runs=[]; begin=time.perf_counter()
    for run in range(1,RUNS+1):
        detected=success=0; bers=[]; psnrs=[]; ssims=[]
        for i,path in enumerate(images,1):
            try:
                with Image.open(path) as source: original=source.convert("RGB")
                attacked=attack(original); decoded,present,schema=model.decode(attacked); exact=bool(present and decoded==expected); authenticated=bool(exact and verify(public,sig))
                detected+=int(bool(present)); success+=int(authenticated); e=payload_ber(decoded,expected); p,s=metrics(original,attacked); bers.append(e); psnrs.append(p); ssims.append(s)
                details.append((run,path.name,bool(present),str(decoded or ""),expected,exact,authenticated,e,p,s,str(schema or ""),""))
                if run==1: attacked.save(ATTACK_DIR/(path.stem+"_JPEG_80_60_40.jpg"),"JPEG",quality=100)
            except Exception as error: details.append((run,path.name,False,"",expected,False,False,1.0,"","","",repr(error)))
        runs.append((detected,success,np.mean(bers),np.mean(psnrs),np.mean(ssims))); print(f"Run {run}/{RUNS}: detected={detected}, authenticated={success}")
    d=np.array([x[0] for x in runs],float); s=np.array([x[1] for x in runs],float); fail=REPORT_TOTAL-s; b=np.array([x[2] for x in runs],float)
    summary=(RUNS,REPORT_TOTAL,d.mean(),d.std(),d.mean()/REPORT_TOTAL,s.mean(),s.std(),s.mean()/REPORT_TOTAL,fail.mean(),fail.std(),b.mean(),b.std(),np.mean([x[3] for x in runs]),np.mean([x[4] for x in runs]),expected)
    write_excel(details,summary); print("Completed in %.2f minutes"%((time.perf_counter()-begin)/60)); print("Excel:",EXCEL_PATH); print("Images:",ATTACK_DIR)
if __name__=="__main__": main()
