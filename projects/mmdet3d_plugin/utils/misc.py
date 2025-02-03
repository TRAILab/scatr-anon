import hashlib
import numpy as np

def hash_tensor(tensor):
    return hash_array(tensor.cpu().detach().numpy())

# https://stackoverflow.com/a/77212976
def hash_array(array):
    int_view = array.view(np.uint8)

    # https://stackoverflow.com/a/26782930
    int_view_cont = int_view.copy(order='C')

    return hashlib.sha1(int_view_cont).hexdigest()