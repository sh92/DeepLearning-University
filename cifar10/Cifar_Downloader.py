import sys
import os
import urllib.request
import tarfile
import zipfile
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def _print_download_progress(count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r- 다운로드 진행 중 : {0:.1%}\n".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()

def maybe_download_and_extract(url, download_dir):
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)

    if not os.path.exists(file_path):
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        file_path, _ = urllib.request.urlretrieve(url=url,filename=file_path,reporthook=_print_download_progress)

        print()
        print("다운로드 완료. 추출 중입니다. \n")

        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)

        print("Done.")
    else:
        print("이미 데이터가 존재합니다.\n")

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def  run():
    name = 'data_batch_1'
    batch = unpickle("data/CIFAR-10/cifar-10-batches-py/{}".format(name))
    trainX = batch[b'data']
    trainY = batch[b'labels']
    for name in ['data_batch_2','data_batch_3','data_batch_4','data_batch_5']:
        batch = unpickle("data/CIFAR-10/cifar-10-batches-py/{}".format(name))
        trainX = np.append(trainX,batch[b'data'],axis=0)
        trainY = np.append(trainY,batch[b'labels'],axis=0)
    test = unpickle("data/CIFAR-10/cifar-10-batches-py/test_batch")
    testX = test[b'data']
    testY = test[b'labels']
    print("npy로 변환 중\n")
    
    trainY = np.expand_dims(trainY,axis=1)
    testY = np.expand_dims(testY,axis=1)
    
    np.save('./np_data/trainX',trainX)
    np.save('./np_data/trainY',trainY)
    np.save('./np_data/testX',testX)
    np.save('./np_data/testY',testY)
    
    print("Enter키를 누르면 종료합니다.")
    input()

if __name__ == "__main__":
    print("=====(주의) numpy가 설치되어 있어야 합니다.=====")
    print("Enter키를 누르면 다운로드를 시작 합니다.")
    input()
    data_path = "data/CIFAR-10/"
    data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    maybe_download_and_extract(url=data_url, download_dir=data_path)
    run()
    
    

