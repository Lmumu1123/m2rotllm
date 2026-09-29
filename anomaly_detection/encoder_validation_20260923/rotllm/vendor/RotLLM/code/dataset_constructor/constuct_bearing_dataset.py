from utils.data.dataset_constructor.bearing.cwru import CWRUCons
from utils.data.dataset_constructor.bearing.dirg import DIRGCons
from utils.data.dataset_constructor.bearing.hit import HITCons
from utils.data.dataset_constructor.bearing.ims import IMSCons
from utils.data.dataset_constructor.bearing.just import JUSTCons
from utils.data.dataset_constructor.bearing.mfpt import MFPTCons
from utils.data.dataset_constructor.bearing.ncepu import NCEPUCons
from utils.data.dataset_constructor.bearing.pu import PUCons
from utils.data.dataset_constructor.bearing.sdust import SDUSTCons
from utils.data.dataset_constructor.bearing.xjtu import XJTUCons

if __name__ == '__main__':
    CWRUCons().process_all_data()
    DIRGCons().process_all_data()
    HITCons().process_all_data()
    IMSCons().process_all_data()
    JUSTCons().process_all_data()
    MFPTCons().process_all_data()
    NCEPUCons().process_all_data()
    PUCons().process_all_data()
    XJTUCons().process_all_data()
    SDUSTCons().process_all_data()

