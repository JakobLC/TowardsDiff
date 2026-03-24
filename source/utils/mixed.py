import torch
import numpy as np
import random
from pathlib import Path
import csv
import os
import json
import jsonlines
import shutil
import datetime
import re
import warnings
import time
import copy
import scipy.ndimage as nd
import matplotlib


class MatplotlibTempBackend():
    def __init__(self,backend):
        self.backend = backend
    def __enter__(self):
        self.old_backend = matplotlib.get_backend()
        matplotlib.use(self.backend)
    def __exit__(self, exc_type, exc_val, exc_tb):
        matplotlib.use(self.old_backend)

def keep_step_rows_and_save(load_name,save_name,max_step=None,max_row_idx=None):
    """Loads a csv file and resaves it as a new file by some criterion of the rows.
    """
    if isinstance(load_name,Path):
        load_name = str(load_name)
    if isinstance(save_name,Path):
        save_name = str(save_name)
    if (max_step is not None) or (max_row_idx is not None):
        loaded = np.loadtxt(load_name,delimiter=",",dtype=str)
        header = loaded[0]
        if max_row_idx is None:
            data = loaded[1:]
        else:
            data = loaded[1:max_row_idx+1]
        if max_step is not None:
            assert "step" in header.tolist(), "step column not found in header"
            step_idx=header.tolist().index("step")
            data = data[data[:,step_idx].astype(float).astype(int)<=max_step]
        loaded = np.concatenate([header[None],data],axis=0)
        np.savetxt(save_name,loaded,delimiter=",",fmt="%s")
    else:
        #simply copy a txt-like file
        with open(load_name,"r") as f:
            lines = f.readlines()
        with open(save_name,"w") as f:
            f.writelines(lines)

def longest_common_substring(str1, str2):
    dp = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]
    max_length = 0
    end_position = 0
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_position = i
            else:
                dp[i][j] = 0
    return str1[end_position - max_length: end_position]

def save_dict_list_to_json(data_list, file_path, append=False):
    assert isinstance(file_path,str), "file_path must be a string"
    if not isinstance(data_list,list):
        data_list = [data_list]
    
    if file_path.endswith(".json"):
        loaded_data = []
        if append:
            if Path(file_path).exists():
                loaded_data = load_json_to_dict_list(file_path)
                if not isinstance(loaded_data,list):
                    loaded_data = [loaded_data]
        data_list = loaded_data + data_list
        with open(file_path, "w") as json_file:
            json.dump(data_list, json_file, indent=4)
    else:
        assert file_path.endswith(".jsonl"), "File path must end with .json or .jsonl"
        mode = "a" if append else "w"
        with jsonlines.open(file_path, mode=mode) as writer:
            for line in data_list:
                writer.write(line)

def load_json_to_dict_list(file_path):
    assert len(file_path)>=5, "File path must end with .json"
    assert file_path[-5:] in ["jsonl",".json"], "File path must end with .json or .jsonl"
    if file_path[-5:] == "jsonl":
        assert len(file_path)>=6, "File path must end with .json or .jsonl"
        assert file_path[-6:]==".jsonl","File path must end with .json or .jsonl"
    if file_path[-5:] == ".json":
        with open(file_path, 'r') as json_file:
            data_list = json.load(json_file)
    elif file_path[-6:] == ".jsonl":
        data_list = []
        with jsonlines.open(file_path) as reader:
            for line in reader:
                data_list.append(line)
    return data_list

class DataloaderIterator():
    """
    Class which takes a pytorch dataloader and enables next() ad infinum and 
    self.partial_epoch gives an iterator which only iterates on a ratio of 
    an epoch 
    """
    def __init__(self,dataloader):
        """ initialize the dataloader wrapper
        Args:
            dataloader (torch.utils.data.dataloader.DataLoader): dataloader to sample from
        """
        self.dataloader = dataloader
        self.iter = iter(self.dataloader)
        self.partial_flag = False
        self.partial_round_err = 0

    def __len__(self):
        return len(self.dataloader)
    
    def reset(self):
        """reset the dataloader iterator"""
        self.iter = iter(self.dataloader)
        self.partial_flag = False
        self.partial_round_err = 0
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.partial_flag:
            if self.partial_counter==self.partial_counter_end:
                self.partial_flag = False
                raise StopIteration
        try:
            batch = next(self.iter)
        except StopIteration:    
            self.iter = iter(self.dataloader)
            batch = next(self.iter)
        if self.partial_flag:
            self.partial_counter += 1
        return batch

    def partial_epoch(self,ratio):
        """returns an iterable stopping after a partial epoch 
        Args:
            ratio (float): positive float denoting the ratio of the epoch.
                           e.g. 0.2 will give 20% of an epoch, 1.5 will
                           give one and a half epoch
        Returns:
            iterable: partial epoch iterable
        """
        self.partial_flag = True
        self.partial_counter_end_unrounded = len(self.dataloader)*ratio+self.partial_round_err
        self.partial_counter_end = int(round(self.partial_counter_end_unrounded))
        self.partial_round_err = self.partial_counter_end_unrounded-self.partial_counter_end
        self.partial_counter = 0
        if self.partial_counter_end==0:
            self.partial_counter_end = 1
        return iter(self)

def didx_from_info(info):
    if isinstance(info,dict):
        return f"{info['dataset_name']}/{info['i']}"
    elif isinstance(info,list):
        assert all([isinstance(info_i,dict) for info_i in info]), "expected all elements of info to be dicts, or info to be a dict"
        return [didx_from_info(info_i) for info_i in info]

def check_keys_are_same(list_of_dicts,verbose=True):
    assert isinstance(list_of_dicts,list), "list_of_dicts must be a list"
    keys = [sorted(list(d.keys())) for d in list_of_dicts]
    if len(keys)==0:
        return True
    else:
        uq_keys = set(sum(keys,[]))
        for k in uq_keys:
            keys_found_k = [int(k in d) for d in keys]
            if not all(keys_found_k):
                if verbose:
                    print(f"Key {k} not found in all dictionaries keys_found_k={keys_found_k}")
                return False
        return True
            
def format_relative_path(path):
    if path is None:
        return path
    return str(Path(path).resolve().relative_to(Path(".").resolve()))

def imagenet_preprocess(x,inv=False,dim=1,maxval=1.0):
    """Normalizes a torch tensor or numpy array with 
    the imagenet mean and std. Can also be used to
    invert the normalization. Assumes the input is
    in the range [0,1]."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    assert x.shape[dim]==3, f"x must have 3 channels in the specified dim={dim}, x.shape: {str(x.shape)}"
    shape = [1 for _ in range(len(x.shape))]
    shape[dim] = 3
    assert torch.is_tensor(x) or isinstance(x,np.ndarray), "x must be a torch tensor or numpy array"
    if torch.is_tensor(x):
        mean = torch.tensor(mean).to(x.device).reshape(shape)
        std = torch.tensor(std).to(x.device).reshape(shape)
    else:
        mean = np.array(mean).reshape(shape)
        std = np.array(std).reshape(shape)
    if inv:
        #y = (x-mean)/std <=> x = y*std + mean
        m = std*maxval
        b = mean*maxval
    else:
        #y = (x-mean)/std = x*1/std - mean/std
        m = 1/std/maxval
        b = -mean/std
    out = x*m+b
    if abs(255-maxval)<1e-6:
        if isinstance(x,np.ndarray):
            out = np.clip(out,0,255).astype(np.uint8)
        else:
            out = torch.clamp(out,0,255).to(torch.uint8)
    return out

def is_infinite_and_not_none(x):
    if x is None:
        return False
    else:
        return torch.isinf(x).any()

def fancy_print_kvs(kvs, atmost_digits=5, s="#"):
        """prints kvs in a nice format like
         |#########|########|
         | key1    | value1 |
              ...
         | keyN    | valueN |
         |#########|########|
        """
        values_print = []
        keys_print = []
        for k,v in kvs.items():
            if isinstance(v,float):
                v = f"{v:.{atmost_digits}g}"
            else:
                v = str(v)
            values_print.append(v)
            keys_print.append(k) 
        max_key_len = max([len(k) for k in keys_print])
        max_value_len = max([len(v) for v in values_print])
        print_str = "\n"
        print_str += "|" + s*(max_key_len+2) + "|" + s*(max_value_len+2) + "|\n"
        for k,v in zip(keys_print,values_print):
            print_str += "| " + k + " "*(max_key_len-len(k)+1) + "| " + v + " "*(max_value_len-len(v)+1) + "|\n"
        print_str += "|" + s*(max_key_len+2) + "|" + s*(max_value_len+2) + "|\n"
        return print_str

def bracket_glob_fix(x):
    return "[[]".join([a.replace("]","[]]") for a in x.split("[")])

def dump_kvs(filename, kvs, sep=","):
    file_exists = os.path.isfile(filename)
    if file_exists:
        with open(filename, 'r', newline='') as file:
            reader = csv.reader(file, delimiter=sep)
            old_headers = next(reader, [])
        new_headers = set(kvs.keys()) - set(old_headers)
        if new_headers:
            header_write = old_headers + sorted(new_headers)
            
            with open(filename, 'r', newline='') as file:
                reader = csv.reader(file, delimiter=sep)
                data = list(reader)
            
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file, delimiter=sep)
                writer.writerow(header_write)  # Write sorted headers
                #remove the old header
                data.pop(0)
                # Modify old lines to have empty values for the new columns
                for line in data:
                    line_dict = dict(zip(old_headers, line))
                    line_dict.update({col: "" for col in new_headers})
                    writer.writerow([line_dict[col] for col in header_write])
            
        else:
            header_write = old_headers
    else:
        # create a file with headers
        header_write = sorted(kvs.keys())
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=sep)
            writer.writerow(header_write)  # Write sorted headers
    
    # Write the key-value pairs to the file, taking into account that only some columns might be present
    kvs_write = {col: kvs[col] if col in kvs else "" for col in header_write}
    with open(filename, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header_write, delimiter=sep)
        writer.writerow(kvs_write)

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    if seed is not None:
        if seed < 0:
            seed = None
    if seed is None:
        np.random.seed()
        seed = np.random.randint(0, 2**16-1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed

def wildcard_match(pattern, text, warning_on_star_in_text=True):
    """
    Perform wildcard pattern matching.

    Parameters:
        pattern (str): The wildcard pattern to match against. '*' matches any character
                      zero or more times.
        text (str): The text to check for a match against the specified pattern.

    Returns:
        bool: True if the text matches the pattern, False otherwise."""
    if '*' in text and warning_on_star_in_text:
        warnings.warn("Wildcard pattern matching with '*' in text is not recommended.")
    pattern = re.escape(pattern)
    pattern = pattern.replace(r'\*', '.*')
    regex = re.compile(pattern)
    return bool(regex.search(text))

def get_time(verbosity=4,sep="-"):
    if verbosity==0:
        s = datetime.datetime.now().strftime('%m-%d')
    elif verbosity==1:
        s = datetime.datetime.now().strftime('%m-%d-%H-%M')
    elif verbosity==2:
        s = datetime.datetime.now().strftime('%y-%m-%d-%H-%M')
    elif verbosity==3:
        s = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    elif verbosity==4:
        s = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    else:
        raise ValueError("Unknown verbosity level: "+str(verbosity))
    if sep!="-":
        s = s.replace("-",sep)
    return s

def format_save_path(args):
    save_path = str(Path("./saves/") / f"{get_time(1)}_{args.model_id}")
    for k,v in args.origin.items():
        if v=="modified_args" and (k not in ["model_id","origin","model_name","save_path"]):
            save_path += f"_({k}={getattr(args,k)})"
    return save_path

def mask_from_imshape(imshape,resize,num_dims=2):
    h,w = imshape[:2]
    new_h,new_w = sam_resize_index(h,w,resize=resize)
    mask = np.zeros((resize,resize),dtype=bool)
    mask[:new_h,:new_w] = True
    for _ in range(num_dims-2):
        mask = mask[None]
    return mask

def sam_resize_index(h,w,resize=64):
    if h>w:
        new_h = resize
        new_w = np.round(w/h*resize).astype(int)
    else:
        new_w = resize
        new_h = np.round(h/w*resize).astype(int)
    return new_h,new_w

def segmentation_gaussian_filter(seg,sigma=1,skip_index=[],skip_spatial=None,padding="constant"):
    assert len(seg.shape)==2 or (len(seg.shape)==3 and seg.shape[-1]==1), f"expected seg to be of shape (H,W) or (H,W,1), found {seg.shape}"
    assert seg.dtype==np.uint8, f"expected seg to be of type np.uint8, found {seg.dtype}"
    if skip_spatial is not None:
        assert skip_spatial.shape==seg.shape, f"skip_spatial.shape={skip_spatial.shape} != seg.shape={seg.shape}"
        ssf = lambda x: np.logical_and(x,np.logical_not(skip_spatial))
    else:
        ssf = lambda x: x
    uq = np.unique(seg)
    best_val = np.zeros_like(seg,dtype=float)
    best_idx = -np.ones_like(seg,dtype=int)
    for i in uq:
        if i in skip_index:
            continue
        val = nd.gaussian_filter(ssf(seg==i).astype(float),sigma=sigma,mode=padding)
        mask = val>best_val
        best_val[mask] = val[mask]
        best_idx[mask] = i
    if any(best_idx.flatten()<0):
        #set all pixels that were not assigned to the closest set pixel class
        _, idx = nd.distance_transform_edt(best_idx<0,return_indices=True)
        best_idx[best_idx<0] = best_idx[tuple(idx[:,best_idx<0])]
        #print("Warning: some pixels were not assigned to any class")
    best_idx = best_idx.astype(np.uint8)
    return best_idx

def postprocess_batch(seg_tensor,seg_kwargs={},overwrite=False,keep_same_type=True,list_of_imshape=None):
    """
    Wrapper for postprocess_list_of_segs that handles many types of 
    batched inputs and returns the same type for the output.
    """
    expected_seg_tensor_msg = "Expected seg_tensor to be a torch.Tensor or np.ndarray or a list of torch.Tensor or np.ndarray"
    bs = len(seg_tensor)
    if list_of_imshape is not None:
        assert isinstance(list_of_imshape,list), "expected list_of_imshape to be a list"
        assert bs>0, "expected bs>0"
        assert len(list_of_imshape)==len(seg_tensor), f"expected len(list_of_imshape)={len(list_of_imshape)} to be equal to len(seg_tensor)={len(seg_tensor)}"
        assert all([isinstance(imshape,(tuple,list)) for imshape in list_of_imshape]), "expected all elements of list_of_imshape to be a tuple or list"
        assert all([len(imshape)>=2 for imshape in list_of_imshape]), "expected all elements of list_of_imshape to have length >=2"

    if torch.is_tensor(seg_tensor):
        input_mode = "torch"
        dtype = seg_tensor.dtype
        device = seg_tensor.device
        transform = lambda x: x.cpu().numpy()
    elif isinstance(seg_tensor,np.ndarray):
        input_mode = "np"
        transform = lambda x: x
    else:
        assert isinstance(seg_tensor,list), expected_seg_tensor_msg+", found "+str(type(seg_tensor))
        if torch.is_tensor(seg_tensor[0]):
            input_mode = "list_of_torch"
            dtype = seg_tensor[0].dtype
            device = seg_tensor[0].device
            transform = lambda x: x.cpu().numpy()
        else:
            assert isinstance(seg_tensor[0],np.ndarray), expected_seg_tensor_msg+", found "+str(type(seg_tensor))
            input_mode = "list_of_np"
            transform = lambda x: x
    num_dims = [len(im.shape) for im in seg_tensor]
    num_dims0 = num_dims[0]
    assert all([nd==num_dims0 for nd in num_dims]), "expected all images to have the same number of dimensions"
    dim_is_trivial = [all([im.shape[j]==1 for im in seg_tensor]) for j in range(num_dims0)]
    trivial_idx = [i for i in range(num_dims0) if dim_is_trivial[i]]
    dim_is_nontrivial = [not dit for dit in dim_is_trivial]
    assert sum(dim_is_nontrivial)==2, "expected exactly 2 non-trivial dimensions, found dim_is_nontrivial="+str(dim_is_nontrivial)
    d1,d2 = [i for i in range(num_dims0) if dim_is_nontrivial[i]]
    resize = max([max(seg.shape) for seg in seg_tensor])
    list_of_segs = []
    crop_slices = []
    for i in range(bs):
        if list_of_imshape is None:
            new_h,new_w = None,None
        else:
            new_h,new_w = sam_resize_index(*list_of_imshape[i],resize=resize)
        crop_slice = [0 for _ in range(num_dims0)]
        crop_slice[d1] = slice(0,new_h)
        crop_slice[d2] = slice(0,new_w)
        crop_slice = tuple(crop_slice)
        list_of_segs.append(transform(seg_tensor[i])[crop_slice])
        crop_slices.append(crop_slice)
    if seg_kwargs.get("mode","")==("min_rel_area"):
        areas = [seg.mean() for seg in list_of_segs]
        area_thresh = max(areas)*seg_kwargs.get("min_area",0.5)
        list_of_segs = [seg if area>area_thresh else np.zeros_like(seg) for seg,area in zip(list_of_segs,areas)]
    else:
        list_of_segs = postprocess_list_of_segs(list_of_segs,seg_kwargs=seg_kwargs,overwrite=overwrite)
    
    if keep_same_type:
        if input_mode=="torch":
            for i in range(bs):
                seg_tensor[i][crop_slices[i]] = torch.tensor(list_of_segs[i],dtype=dtype,device=device)
        elif input_mode=="np":
            for i in range(bs):
                seg_tensor[i][crop_slices[i]] = list_of_segs[i]
        elif input_mode=="list_of_torch":
            seg_tensor = [torch.tensor(np.expand_dims(seg,trivial_idx),dtype=dtype,device=device) for seg in list_of_segs]
        else:
            seg_tensor = [np.expand_dims(seg,trivial_idx) for seg in list_of_segs]
    else:
        seg_tensor = list_of_segs
    return seg_tensor

def postprocess_list_of_segs(list_of_segs,seg_kwargs={},overwrite=False):
    out = []
    for seg in list_of_segs:
        out.append(postprocess_seg(seg,**seg_kwargs,overwrite=overwrite))
    return out

def postprocess_seg(seg,
                    mode="gauss_survive",
                    replace_with="nearest",
                    num_objects=8,
                    min_area=0.005,
                    sigma=0.001,
                    overwrite=False):
    """
    Postprocess a segmentation by removing pixels of typically small objects or noise.

    Args:
    seg: np.ndarray, shape (H,W) or (H,W,1), dtype np.uint8
        The segmentation to postprocess.
    mode: str, one of ["num_objects", "min_area", "gauss_raw", "gauss_survive"]
        The mode to use for postprocessing where:
        - "num_objects": remove all but the largest `num_objects` objects
        - "min_area": remove all objects with smaller relative area smaller than 
            `min_area`
        - "gauss_raw": apply a gaussian filter to the onehot of the segmentation
        - "gauss_survive": apply a gaussian filter to the onehot of the segmentation
            and keep the original segmentation for objects that survive the filter
    replace_with: str, one of ["gauss", "new", "nearest"]
        The method to use when replacing pixels of removed objects. Where:
        - "gauss": replace with the result of the gaussian filter
        - "new": replace with a unique label not found in the objects that were kept
        - "nearest": replace with the label of the nearest object from a distance
            transform
    num_objects: int
        The number of objects to keep if `mode` is "num_objects".
    min_area: float
        The minimum relative area of an object to keep if `mode` is "min_area".
    sigma: float
        The sigma of the gaussian filter

    Returns:
    np.ndarray, shape (H,W) or (H,W,1), dtype np.uint8
        The postprocessed segmentation.
    """
    assert mode in ["num_objects", "min_area", "gauss_raw", "gauss_survive","min_rel_area"], f"expected mode to be one of ['num_objects', 'min_area', 'gauss_raw', 'gauss_survive'], found {mode}"
    assert replace_with in ["gauss", "new", "nearest"], f"expected replace_with to be one of ['gauss', 'new', 'nearest'], found {replace_with}"
    if torch.is_tensor(seg):
        was_torch = True
        seg = seg.cpu().numpy()
    else:
        was_torch = False
    assert isinstance(seg,np.ndarray), "expected seg to be an np.ndarray"
    assert seg.dtype==np.uint8
    if mode!="min_rel_area":
        assert len(seg.shape)==2 or (len(seg.shape)==3 and seg.shape[-1]==1), f"expected seg to be of shape (H,W) or (H,W,1), found {seg.shape}"
    else:
        assert len(seg.shape)==3 and seg.shape[-1]>1, "expected seg to be of shape (H,W,C) with C>1, found "+str(seg.shape)
        if seg.shape[2]>seg.shape[0] or seg.shape[2]>seg.shape[1]:
            warnings.warn("Expected the channel dimension to be smaller and last than spacial dims. Found shape: "+str(seg.shape))
    if not overwrite:
        seg = seg.copy()
    gauss_seg = None
    sigma_in_pixels = sigma*np.sqrt(np.prod(seg.shape))
    remove_mask = None
    if mode=="num_objects":
        assert num_objects>0 and isinstance(num_objects,int), "num_objects must be a positive integer. found: "+str(num_objects)
        uq, counts = np.unique(seg.flatten(),return_counts=True)
        if len(uq)>num_objects:
            remove_labels = uq[np.argsort(counts)[:-num_objects]]
            remove_mask = np.isin(seg,remove_labels)
    elif mode=="min_area":
        uq, counts = np.unique(seg,return_counts=True)
        area = counts/seg.size
        if any(area<min_area):
            remove_labels = uq[area<min_area]
            remove_mask = np.isin(seg,remove_labels)
    elif mode=="gauss_raw":
        seg = segmentation_gaussian_filter(seg,sigma=sigma_in_pixels)
    elif mode=="gauss_survive":
        gauss_seg = segmentation_gaussian_filter(seg,sigma=sigma_in_pixels)
        uq = np.unique(gauss_seg)
        remove_mask = np.logical_not(np.isin(seg,uq))
    elif mode=="min_rel_area":
        areas = seg.mean(axis=(0,1))
        area_thresh = max(areas)*min_area
        seg_new = []
        for i in range(len(areas)):
            seg_new.append(seg[:,:,i] if areas[i]>area_thresh else np.zeros_like(seg[:,:,i]))
        seg = np.stack(seg_new,axis=-1)

    if np.all(remove_mask):
        return np.zeros_like(seg)
    replace_vals = None
    if remove_mask is not None:
        if replace_with=="nearest":
            idx_of_nn = nd.distance_transform_edt(remove_mask,return_indices=True)[1]
            replace_vals = seg[tuple(idx_of_nn)]
        elif replace_with=="new":
            uq = np.unique(seg[np.logical_not(remove_mask)])
            first_idx_not_in_uq = [i for i in range(len(uq)+1) if i not in uq][0]
            replace_vals = np.ones_like(seg)*first_idx_not_in_uq
        elif replace_with=="gauss":
            replace_vals = segmentation_gaussian_filter(seg,sigma=sigma_in_pixels,skip_spatial=remove_mask)
        seg[remove_mask] = replace_vals[remove_mask]
    if was_torch:
        seg = torch.tensor(seg)
    return seg

def get_padding_slices(x,shape):
    assert len(x.shape)>=2, "expected at least 2 dimensions in x"
    assert x.shape[-1]==x.shape[-2], "expected a square image, found "+str(x.shape)
    assert len(shape)>=2, "expected len(shape)>=2, found "+str(shape)
    new_h,new_w = sam_resize_index(*shape[:2],resize=x.shape[-1])
    slices = [slice(None) for _ in range(len(x.shape)-2)]
    if new_h==x.shape[-2] and new_w==x.shape[-1]:
        slices += [slice(new_h,new_h),slice(new_w,new_w)]
    elif new_w==x.shape[-1]:
        slices += [slice(new_h,x.shape[-2]),slice(None)]
    elif new_h==x.shape[-2]:
        slices += [slice(None),slice(new_w,x.shape[-1])]
    else:
        raise ValueError(f"Expected at least one of new_h or new_w to be equal to x.shape[-2] or x.shape[-1] with SAM crops. Found new_h={new_h} and new_w={new_w} for x.shape={x.shape}")
    return slices

def apply_mask(x,mask,is_shape=True):
    assert len(x.shape)>=2, "expected at least 2 dimensions in x"
    assert x.shape[-1]==x.shape[-2], "expected a square image, found "+str(x.shape)
    if is_shape:
        assert len(mask)>=2, "expected len(mask)>=2 when is_shape=True"
        new_h,new_w = sam_resize_index(*mask[:2],resize=x.shape[-1])
    else:
        #use the bbox of nonzero values in mask
        assert len(x.shape)>=2 and len(mask.shape)>=2, "expected at least 2 dimensions in x and mask"
        assert x.shape[-2]==mask.shape[-2] and x.shape[-1]==mask.shape[-1], "expected x.shape[-2:] to be equal to mask.shape[-2:]"
        
        new_h,new_w = max_nonzero_per_dim(mask)[-2:]
    slices = [slice(None) for _ in range(len(x.shape)-2)]+[slice(0,new_h),slice(0,new_w)]
    return x[tuple(slices)]

def torch_any_multiple(x,axis):
    out = x
    for dim in axis:
        out = out.any(dim=dim)
    return out

def max_nonzero_per_dim(x,add_one=True):
    if isinstance(x,np.ndarray):
        f = lambda x,dim: np.nonzero(np.any(x,axis=tuple([i for i in range(x.ndim) if i!=dim])))[0].tolist()
    else:
        f = lambda x,dim: torch.nonzero(torch_any_multiple(x,axis=[i for i in range(x.ndim) if i!=dim])).flatten().tolist()

    nnz = [f(x,i) for i in range(x.ndim)]
    print(nnz)
    nnz = [max([0]+v)+int(add_one) for v in nnz]
    return nnz
    
def to_dev(item,device="cuda"):
    if torch.is_tensor(item):
        return item.to(device)
    elif isinstance(item,list):
        return [to_dev(i,device) for i in item]
    elif item is None:
        return None
    else:
        raise ValueError(f"Unknown type: {type(item)}. Expected list of torch.tensor or None")

def model_arg_is_trivial(model_arg_k):
    out = False
    if model_arg_k is None:
        out = True
    elif isinstance(model_arg_k,list):
        if len(model_arg_k)==0:
            out = True
        elif all([item is None for item in model_arg_k]):
            out = True
    return out
    
def nice_split(s,split_s=",",remove_empty_str=True):
    assert isinstance(s,str), "expected s to be a string"
    assert isinstance(split_s,str), "expected split_s to be a string"
    if len(s)==0:
        out = []
    else:
        out = s.split(split_s)
    if remove_empty_str:
        out = [item for item in out if len(item)>0]
    return out

def fix_clip_matrix_in_state_dict(ckpt_model,model):
    if "vit.class_names_embed.0.weight" in ckpt_model.keys():
        if ckpt_model["vit.class_names_embed.0.weight"].shape[0]!=model.vit.class_names_embed[0].weight.shape[0]:
            print("WARNING: class_names_embed weight shape mismatch. Ignoring.")
            ckpt_model["vit.class_names_embed.0.weight"] = model.vit.class_names_embed[0].weight
    return ckpt_model

def format_model_kwargs(model_kwargs,del_none=True,dev="cuda",list_instead=False):
    """Formats a kwarg dictionary with list arguments as 
    a tensor on the specified device"""
    bs = None
    for k in model_kwargs.keys():
        if not model_arg_is_trivial(model_kwargs[k]):
            if bs is None:
                bs = len(model_kwargs[k])
            else:
                assert bs==len(model_kwargs[k]), f"expected same bs. Found {bs} and {len(model_kwargs[k])} for {k}"
            model_kwargs[k] = unet_kwarg_to_tensor(model_kwargs[k],key=k,dev=dev,list_instead=list_instead)
        else:
            model_kwargs[k] = None

    if del_none:
        for k in list(model_kwargs.keys()):
            if model_kwargs[k] is None:
                del model_kwargs[k]
    return model_kwargs

def unet_kwarg_to_tensor(kwarg,key=None,non_tensor_exception_keys=["class_names"],dev=None,list_instead=False):
    key_exception = False
    if key is not None:
        if key in non_tensor_exception_keys:
            key_exception = True
    if kwarg is None:
        pass
    elif torch.is_tensor(kwarg):
        pass
    elif key_exception:
        crit = [isinstance(item,(str,tuple,int,torch.Tensor,list)) or (item is None) for item in kwarg]
        assert all(crit), f"If kwarg for exception keys is a list, then all elements must be str, tuple, int, or torch.Tensor. kwarg={kwarg}"
        if model_arg_is_trivial(kwarg):
            kwarg = None
        else:
            kwarg = [[] if item is None else item for item in kwarg]
    elif isinstance(kwarg, list):
        assert all([(isinstance(kw, torch.Tensor) or kw is None) for kw in kwarg]), f"If kwarg is a list, all elements must be torch.Tensor or None. kwarg={kwarg}"
        if all([kw is None for kw in kwarg]): #also return true for empty list
            kwarg = None
        elif all([isinstance(kw, torch.Tensor) for kw in kwarg]):
            if list_instead:
                pass
            else:
                kwarg = torch.stack(kwarg)
        else:
            bs = len(kwarg)
            shapes = [kw.shape for kw in kwarg if kw is not None]
            s0 = [i for i in range(bs) if kwarg[i] is not None][0]
            assert all([s==shapes[0] for s in shapes]), f"If kwarg is a list, all tensors must have the same shape. kwarg={kwarg}"
            if list_instead:
                full_kwarg = [None for _ in range(bs)]
            else:
                full_kwarg = torch.zeros((bs,)+shapes[0],
                                     dtype=kwarg[s0].dtype,
                                     device=kwarg[s0].device)
            for i in range(bs):
                if kwarg[i] is not None:
                    full_kwarg[i] = kwarg[i]
            kwarg = full_kwarg
    else:
        raise ValueError(f"kwarg={kwarg} is not a valid type. must be None, torch.Tensor, or list of torch.Tensor/None")

    if (dev is not None) and torch.is_tensor(kwarg):
        kwarg = kwarg.to(dev)
    elif (dev is not None) and isinstance(kwarg, list):
        assert all([torch.is_tensor(kw) for kw in kwarg if kw is not None]), f"expected all elements of kwarg to be torch.Tensors, found {kwarg}"
        kwarg = [(kw.to(dev) if kw is not None else None) for kw in kwarg]
    return kwarg

def assert_one_to_one_list_of_str(list1,list2):
    assert isinstance(list1,list) and isinstance(list2,list), "Expected list, found: "+str(type(list1))+" and "+str(type(list2))
    for k in list1+list2:
        assert isinstance(k,str), "Expected str, found: "+str(type(k))
        assert k in list1, "Expected "+k+" from list2 to be in list1="+str(list1)
        assert k in list2, "Expected "+k+" from list1 to be in list2="+str(list2)
