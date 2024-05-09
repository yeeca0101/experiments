import os
import requests
import shutil

def bar_custom(current, total):
    progress = f"Downloading: {current*100//total}% [{current} / {total}] bytes"
    print(progress, end='\r')

def download_file(url, out_path, verify=True):
    with requests.get(url, stream=True, verify=verify) as r:
        r.raise_for_status()
        total_length = int(r.headers.get('content-length'))
        current_length = 0
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    current_length += len(chunk)
                    bar_custom(current_length, total_length)
    print()

def download_imagenet(root='./data', verify_ssl=True):
    """
    Download ImageNet validation set
    :param root: Root directory for downloading ImageNet
    :param verify_ssl: Whether to verify SSL certificates
    :return:
    """

    val_url = 'http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_val.tar'
    devkit_url = 'http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_devkit_t12.tar.gz'

    print("Download...")
    os.makedirs(root, exist_ok=True)
    val_path = os.path.join(root, 'ILSVRC2012_img_val.tar')
    devkit_path = os.path.join(root, 'ILSVRC2012_devkit_t12.tar.gz')

    download_file(val_url, val_path, verify=verify_ssl)
    download_file(devkit_url, devkit_path, verify=verify_ssl)

    print('done!')

# Example usage:
download_imagenet(root='./data', verify_ssl=False)
