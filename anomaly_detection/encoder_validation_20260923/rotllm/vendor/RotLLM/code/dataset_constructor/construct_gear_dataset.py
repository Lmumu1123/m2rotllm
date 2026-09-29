from utils.data.dataset_constructor.gearbox.xjtu import XJTUCons
from utils.data.dataset_constructor.gearbox.seu import SEUCons
from utils.data.dataset_constructor.gearbox.sdust import SDUSTCons

if __name__ == '__main__':
    XJTUCons().process_all_data()
    SEUCons().process_all_data()
    SDUSTCons().process_all_data()
