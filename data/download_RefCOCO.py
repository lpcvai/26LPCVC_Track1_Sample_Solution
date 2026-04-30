from utils.refcoco_utils import get_refcoco_dataset

if __name__ == '__main__':
    data = get_refcoco_dataset()
    data.save_to_disk("data/annotations/refcoco-unc")

