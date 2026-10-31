"""Package only this candidate source, tests, and public example configuration."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    target=parser.parse_args().output.resolve()
    names=["README.md","OPERATIONS.md","SUBMISSION.md","PROVENANCE.md","LICENSE",
           "requirements.txt","Dockerfile","compose.yaml","Caddyfile",".dockerignore",
           ".gitignore",".gitattributes","env.example","start_local.ps1","preflight.py","package_source.py"]
    files={name:root/name for name in names}
    for folder in ["adaptive_evidence","tests"]:
        for path in sorted((root/folder).glob("*.py")):
            files[path.relative_to(root).as_posix()]=path
    manifest={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
    target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_DEFLATED) as bundle:
        for name,path in files.items():bundle.write(path,name)
        bundle.writestr("SHA256SUMS.json",json.dumps(manifest,indent=2))
    with zipfile.ZipFile(target) as bundle:
        if bundle.testzip() is not None:raise RuntimeError("Archive checksum failed")
        for name,expected in manifest.items():
            if hashlib.sha256(bundle.read(name)).hexdigest()!=expected:raise RuntimeError("Source checksum mismatch")
    report={"version":"0.32.0","archive":str(target),"source_files":len(files),
            "bytes":target.stat().st_size,"sha256":hashlib.sha256(target.read_bytes()).hexdigest(),
            "verified":True,"official_submission":False,"contains_datasets_or_credentials":False}
    target.with_suffix(".manifest.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    (root/"SHA256SUMS.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
