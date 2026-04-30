from utils.refcoco_utils import download_refcoco_dataset

if __name__ == '__main__':
    data = download_refcoco_dataset()
    data.save_to_disk("data/annotations/refcoco-unc")

