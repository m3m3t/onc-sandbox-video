import tarfile
import importlib
import os

def install_and_import(module,package):
    try:
        importlib.import_module(module)
    except ImportError:
        import pip
        if pip.__version__.startswith('10'):
            from pip._internal import main as pipmain
        else:
            from pip import main as pipmain 
        
        pipmain(['install', '--target=.', package])
    finally:
        globals()[module] = importlib.import_module(module)

def untar(tar_fn, outPath, filetype="bmp"):
    images =[] 
    if tarfile.is_tarfile(tar_fn):
        with tarfile.open(tar_fn) as tar:
            members = [ m for m in tar.getmembers() if m.name.endswith(filetype) ]
            images = [ m.name for m in members ]
            extracted = tar.extractall(path=outPath, members=members)
    return images

def tar(tar_fn, outPath, source_dir):
    with tarfile.open(tar_fn, "w:gz") as tar:
        tar.add(os.path.join(outPath, source_dir), arcname=source_dir)
