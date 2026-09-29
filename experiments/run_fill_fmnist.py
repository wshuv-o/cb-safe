import _bootstrap, os, subprocess, sys
CONFIGS=[("trimmed",1),("reputation",4),("hybrid",4)]
FS=[0.2,0.3,0.1]; SEEDS=[0,1,2]
here=os.path.dirname(os.path.abspath(__file__))
jobs=[(a,ov,f,s) for (a,ov) in CONFIGS for f in FS for s in SEEDS]
if "reverse" in sys.argv: jobs=list(reversed(jobs))
for a,ov,f,s in jobs:
    suf=f"_ov{ov}" if ov!=1 else ""
    name=f"robust_signflip_{a}{suf}_f{int(f*100):02d}_c3_s{s}.csv"
    if os.path.exists(os.path.join(_bootstrap.RESULTS,"fmnist",name)):
        print("skip",name,flush=True); continue
    print("run",a,f"ov{ov}",f,s,flush=True)
    subprocess.call([sys.executable,os.path.join(here,"run_robustness.py"),
        "--attack","signflip","--f",str(f),"--aggregator",a,"--overlap",str(ov),
        "--dataset","fmnist","--rounds","30","--seed",str(s),"--cluster-size","3"])
print("FMNIST FILL COMPLETE",flush=True)
