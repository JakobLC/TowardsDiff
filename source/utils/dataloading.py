import torch
import numpy as np
from scipy.ndimage import center_of_mass
from PIL import Image
from pathlib import Path
import os
import albumentations as A
import cv2
import copy
from argparse import Namespace
import warnings
#add source folder if it is not already in PATH
import sys
if not str(Path(__file__).parent.parent) in sys.path:
    sys.path.append(str(Path(__file__).parent.parent))
from source.utils.mixed import (load_json_to_dict_list,
                                DataloaderIterator)
from source.utils.argparsing import load_existing_args,TieredParser

def custom_collate_with_info(original_batch):
    n = len(original_batch[0])
    normal_batch = []
    for i in range(n):
        list_of_items = [item[i] for item in original_batch]
        if i+1==n:
            info = list_of_items
        else:
            normal_batch.append(torch.stack(list_of_items,axis=0))
    return *normal_batch,info

def binary_dist(b1,b2):
    """counts the number of bits that are different between two binary strings (iterables)"""
    return sum([1 for i in range(len(b1)) if b1[i]!=b2[i]])

def different_ordering(n_bits):
    order = max_diff_sequence(n_bits)
    h = int(np.round(len(order)**0.5))
    #reverse every second row
    for i in range(h):
        if i % 2 == 1:
            order[i*h:(i+1)*h] = order[i*h:(i+1)*h][::-1]
    return order

def max_diff_sequence(n_bits):
    bits = [np.binary_repr(i,width=n_bits) for i in range(2**n_bits)]
    all_bits = list(range(2**n_bits))
    sequence = [all_bits.pop(0)]
    while all_bits:
        last = bits[sequence[-1]]
        next_idx = max(all_bits, key=lambda x: binary_dist(last, bits[x]))
        sequence.append(next_idx)
        all_bits.remove(next_idx)
    return sequence

def gray_code(n):
    if n == 0:
        return ['']
    prev = gray_code(n - 1)
    return ['0' + code for code in prev] + ['1' + code for code in reversed(prev)]

def similar_ordering(n_bits):
    order = [int(code, 2) for code in gray_code(n_bits)]
    h = int(np.round(len(order)**0.5))
    #reverse every second row
    for i in range(h):
        if i % 2 == 1:
            order[i*h:(i+1)*h] = order[i*h:(i+1)*h][::-1]
    return order

class LocationAwarePalette:
    def __init__(self,
                max_num_classes,
                image_size,
                padding_idx=-1,
                mode="similar",
                largest_first=False,
                random_seed=0):
        valid_modes = ["random","similar","different","range"]
        assert mode in valid_modes, "mode must be one of "+str(valid_modes)+", got "+mode
        if mode in ["similar","different"]:
            assert (np.log2(max_num_classes)/2).is_integer(), "max_num_classes must be an even power of 2 (:=2^(2k)), got "+str(max_num_classes)
        else:
            assert np.sqrt(max_num_classes).is_integer(), "max_num_classes must be a perfect square, got "+str(max_num_classes)    
        self.max_num_classes = max_num_classes
        self.num_bits = int(np.log2(max_num_classes))
        self.sidelength = int(np.sqrt(max_num_classes))
        self.image_size = image_size
        self.padding_idx = padding_idx
        self.largest_first = largest_first
        if mode=="random":
            rng = np.random.default_rng(random_seed)
            self.order = rng.permutation(max_num_classes)
        elif mode=="similar":
            self.order = similar_ordering(self.num_bits)
        elif mode=="different":
            self.order = different_ordering(self.num_bits)
        elif mode=="range":
            self.order = np.arange(max_num_classes)
        self.order = np.array(self.order).reshape((self.sidelength,self.sidelength))

    def apply_lap(self, labels):
        mapping = self.mapping_from_labels(labels)
        mapping[self.padding_idx] = self.padding_idx  # ensure padding index is preserved
        #apply with vectorized operation
        labels_flat = labels.flatten()
        mapped_labels = np.vectorize(mapping.get)(labels_flat)
        return mapped_labels.reshape(labels.shape)

    def get_centers(self,labels,relative=True):
        assert len(labels.shape)==2, "labels must be of shape (H,W), got "+str(labels.shape)
        assert labels.shape[0]==labels.shape[1], "labels must be square, got "+str(labels.shape)
        assert labels.shape[0]==self.image_size, "labels must be of shape (image_size,image_size), got "+str(labels.shape)+" for image_size="+str(self.image_size)
        uq_labels,uq_counts = np.unique(labels, return_counts=True)
        uq_counts = uq_counts[self.padding_idx!=uq_labels]  # remove padding index
        uq_labels = uq_labels[self.padding_idx!=uq_labels]  # remove padding index
        centers = center_of_mass(np.ones_like(labels), labels, uq_labels)
        centers = np.array(centers)
        #map -0.5 to 0 and image_size+0.5 to 1
        if relative:
            centers = self.to_rel_coords(centers)
        return centers,uq_labels,uq_counts

    def to_rel_coords(self,centers):
        return (centers + 0.5) / self.image_size

    def to_abs_coords(self,centers):
        return (centers * self.image_size) - 0.5

    def mapping_from_labels(self,labels):
        return self.mapping_from_centers(*self.get_centers(labels))
    
    def mapping_from_centers(self,centers,indices,counts):
        assert len(centers.shape)==2, "centers must be of shape (N,2), got "+str(centers.shape)
        assert centers.shape[1]==2, "centers must be of shape (N,2), got "+str(centers.shape)
        assert centers.min()>= 0, "Centers must be in the range [0,1] for relative coordinates, got "+str(centers.min())
        assert centers.max()<= 1, "Centers must be in the range [0,1] for relative coordinates, got "+str(centers.max())
        assert len(centers)<=self.max_num_classes, "centers must not exceed max_num_classes, got "+str(len(centers))+" for max_num_classes="+str(self.max_num_classes)
        centers = centers * self.sidelength - 0.5
        centers_rounded = np.round(centers).astype(int)
        n = len(centers)
        sampled = self.order[centers_rounded[:,0],centers_rounded[:,1]]
        #handle overlaps:
        if self.largest_first or n>len(np.unique(sampled)):
            #loop through all centers that are not unique. Ignore the first one.
            #assign all subsequent centers to the nearest unused class in L2 distance
            mask = np.zeros_like(self.order, dtype=bool)
            sampled = [None for _ in range(n)]
            Y, X = np.indices(self.order.shape)
            nrange = np.argsort(counts)[::-1] if self.largest_first else np.arange(n)
            for i in nrange:
                if mask[tuple(centers_rounded[i])]:
                    #find the nearest unused class and use it instead of the current one
                    distances = np.sqrt((X - centers[i,1])**2 + (Y - centers[i,0])**2)
                    unused_classes = np.where(~mask.flatten())[0]
                    nearest_unused_class = unused_classes[np.argmin(distances.flatten()[unused_classes])]
                    sampled[i] =  self.order.flatten()[nearest_unused_class]
                    nearest_unused_class_2d = np.unravel_index(nearest_unused_class, self.order.shape)
                    mask[nearest_unused_class_2d] = True
                else:
                    sampled[i] = self.order[tuple(centers_rounded[i])]
                    mask[tuple(centers_rounded[i])] = True    

        mapping = {k: v for k,v in zip(indices,sampled)}
        mapping[self.padding_idx] = self.padding_idx  # ensure padding index is preserved
        return mapping

class EntityDataset(torch.utils.data.Dataset):
    def __init__(self,split="train",
                      image_size=128,
                      datasets="entity",
                      min_rel_class_area=0.0,
                      max_num_classes=8,
                      map_excess_classes_to="largest",
                      shuffle_labels=True,
                      shuffle_zero=True,
                      data_root=None,
                      geo_aug_p=0.3,
                      padding_idx=-1,
                      imagenet_norm=True,
                      aug_override=None,
                      global_p=1.0,
                      lap_mode="none",
                      is_rgb=False,
                      lap_seed=0):
        self.is_rgb = is_rgb
        self.lap_mode = lap_mode
        self.global_p = global_p
        if self.lap_mode!="none":
            recognized_modes = ["random","similar","different","range"]
            recognized_modes += [f"{m}_largest" for m in recognized_modes]
            assert self.lap_mode in recognized_modes, f"lap_mode={self.lap_mode} not recognized. Must be one of {recognized_modes}"
            assert not shuffle_labels, "shuffle_labels must be False when using LocationAwarePalette"
            if self.is_rgb:
                assert max_num_classes <= 121, "max_num_classes must be <= 121 for RGB encoding, got "+str(max_num_classes)
            else:
                n_bits_half = np.log2(max_num_classes)
                assert np.round(n_bits_half) == n_bits_half, "max_num_classes must be an even power of 2, got "+str(max_num_classes)
                assert n_bits_half//2 == n_bits_half/2, "max_num_classes must be an even power of 2, got "+str(max_num_classes)
            self.LAP = LocationAwarePalette(max_num_classes=max_num_classes,
                                            image_size=image_size,
                                            padding_idx=padding_idx,
                                            mode=self.lap_mode.replace("_largest",""),
                                            largest_first="largest" in self.lap_mode,
                                            random_seed=lap_seed)
        self.geo_aug_p = geo_aug_p
        if data_root is None:
            data_root = str(Path(__file__).parent.parent.parent / "data")
        self.data_root = data_root
        self.image_size = image_size
        self.min_rel_class_area = min_rel_class_area
        self.max_num_classes = max_num_classes
        self.datasets = datasets
        self.padding_idx = padding_idx
        self.gen_mode = False
        self.sam_aug_small = get_sam_aug(image_size,padval=padding_idx, imagenet_norm_p=float(imagenet_norm))
        assert map_excess_classes_to in ["largest","random_different","random_same","zero","same","nearest_expensive"]
        self.map_excess_classes_to = map_excess_classes_to
        self.shuffle_labels = shuffle_labels
        self.shuffle_zero = shuffle_zero
        self.downscale_thresholding_factor = 3

        dataset_check = datasets
        if isinstance(dataset_check,list):
            dataset_check = ",".join(dataset_check)
        if dataset_check is None:
            dataset_check = ""
        dataset_check = str(dataset_check).strip().lower()
        if dataset_check != "entity":
            raise NotImplementedError(
                f"Only data/entity is supported. Received datasets={datasets}."
            )

        self.dataset_name = "entity"
        entity_root = Path(self.data_root) / self.dataset_name
        if not entity_root.exists():
            raise NotImplementedError(f"Only data/entity is supported. Missing folder: {entity_root}")

        if split in ["train","vali","test","all"]:
            split = {"train": 0,"vali": 1, "test": 2, "all": 3}[split]
        assert split in list(range(-1,4)), "invalid split input. must be one of [0,1,2,3] or ['train','vali','test','all'], found "+str(split)
        self.split = split
        
        print("processing dataset:", self.dataset_name)
        items = self.get_entity_items()
        if len(items) == 0:
            warnings.warn("no data in dataset entity for split " + str(split))

        if aug_override is not None:
            assert isinstance(aug_override,bool), "aug_override must be a boolean or None"
            aug = aug_override
        else:
            aug = (split==0)

        # Entity is a natural image dataset, so use picture augmentations.
        self.augment_fn = get_augmentation("pictures",
                                           s=self.image_size,
                                           train=aug,
                                           global_p=self.global_p,
                                           geo_aug_p=self.geo_aug_p)

        self.items = items
        self.length = len(self.items)
        self.didx_to_item_idx = {f"{self.dataset_name}/{item['i']}": idx for idx, item in enumerate(self.items)}
        self.dataset_weights = np.ones(self.length, dtype=float)
        self.len_per_dataset = {self.dataset_name: self.length}
        self.dataset_to_label = {"none": 0, self.dataset_name: 1}

    def get_entity_items(self):
        dataset_name = "entity"
        dataset_root = Path(self.data_root) / dataset_name
        train_json_path = dataset_root / "entityseg_train_lr.json"
        val_json_path = dataset_root / "entityseg_val_lr.json"
        val_txt_path = dataset_root / "val_ims.txt"
        test_txt_path = dataset_root / "test_ims.txt"

        missing = [
            p for p in [train_json_path, val_json_path, val_txt_path, test_txt_path]
            if not p.exists()
        ]
        if len(missing) > 0:
            missing_str = ", ".join([str(p) for p in missing])
            raise FileNotFoundError(
                "Missing entity files: " + missing_str + ". Run data/entity/process_entity.py first."
            )

        train_loaded = load_json_to_dict_list(str(train_json_path))
        val_loaded = load_json_to_dict_list(str(val_json_path))

        # Build a unified image registry keyed by relative path.
        image_registry = {}
        for native_split, loaded in [(0, train_loaded), (2, val_loaded)]:
            for im in loaded["images"]:
                fn = im["file_name"]
                image_registry[fn] = {
                    "fn": fn,
                    "i": None,
                    "imshape": [int(im["height"]), int(im["width"]), 3],
                    "native_split": native_split,
                }

        all_fns = sorted(image_registry.keys())
        for i, fn in enumerate(all_fns):
            image_registry[fn]["i"] = i

        def read_set(path):
            with open(path, "r", encoding="utf-8") as f:
                return set([line.strip() for line in f.readlines() if len(line.strip()) > 0])

        val_set = read_set(val_txt_path)
        test_set = read_set(test_txt_path)

        all_set = set(all_fns)
        unknown_val = val_set - all_set
        unknown_test = test_set - all_set
        if len(unknown_val) > 0 or len(unknown_test) > 0:
            raise ValueError("Split files contain unknown filenames.")
        if len(val_set.intersection(test_set)) > 0:
            raise ValueError("val_ims.txt and test_ims.txt overlap.")

        train_set = all_set - val_set - test_set
        split_to_fns = {
            0: train_set,
            1: val_set,
            2: test_set,
            3: all_set,
        }
        split_idx = self.split
        target_fns = split_to_fns[split_idx]

        items = []
        for fn in all_fns:
            if fn not in target_fns:
                continue
            p = Path(fn)
            if len(p.parts) < 2:
                raise ValueError(f"Expected file_name to include folder and filename, got: {fn}")
            image_folder = p.parts[0]
            mask_folder = image_folder + "_masks"
            label_rel = str(Path(mask_folder) / f"{p.stem}_mask.png")
            label_abs = dataset_root / label_rel
            if not label_abs.exists():
                raise FileNotFoundError(
                    f"Missing mask file for {fn}: {label_abs}. Run data/entity/process_entity.py first."
                )

            info = image_registry[fn]
            item = {
                "dataset_name": dataset_name,
                "i": info["i"],
                "fn": fn,
                "split_idx": 1 if fn in val_set else (2 if fn in test_set else 0),
                "imshape": info["imshape"],
                "classes": [0],
                "class_counts": [int(info["imshape"][0] * info["imshape"][1])],
                "image_path": fn,
                "label_path": label_rel,
            }
            items.append(item)
        return items

    def __len__(self):
        return self.length
    
    def get_sampler(self,seed=None):
        if seed is None:
            generator = None
        else:
            generator = torch.Generator().manual_seed(seed)
        p = self.dataset_weights
        return torch.utils.data.WeightedRandomSampler(p,num_samples=len(self),replacement=True,generator=generator)

    def get_gen_dataset_sampler(self,datasets,seed=None):
        dataset_check = datasets
        if isinstance(dataset_check,list):
            dataset_check = ",".join(dataset_check)
        dataset_check = str(dataset_check).strip().lower()
        if dataset_check != "entity":
            raise NotImplementedError(f"Only data/entity is supported. Received datasets={datasets}.")
        if seed is None:
            generator = None
        else:
            generator = torch.Generator().manual_seed(seed)
        p = self.dataset_weights
        return torch.utils.data.WeightedRandomSampler(p,num_samples=len(self),replacement=False,generator=generator)

    def convert_to_idx(self,list_of_things):
        """
        Converts a list of things to a list of indices. The items in
        the list should either be:
         - a list of integer indices (where we only check that the indices are valid)
         - a list of info dicts with the fields "dataset_name" and "i"
         - a list of strings formatted like '{dataset_name}/{i}'
        Returns a list of integer indices and checks they are valid
        """
        assert isinstance(list_of_things,list)
        if len(list_of_things)==0: return []
            
        item0 = list_of_things[0]
        if isinstance(item0,int):
            list_of_things2 = list_of_things
        elif isinstance(item0,dict):
            assert "dataset_name" in item0 and "i" in item0, "item0 must be a dict with the fields 'dataset_name' and 'i'"
            d_vec = [item["dataset_name"] for item in list_of_things]
            i_vec = [item["i"] for item in list_of_things]
        elif isinstance(item0,str):
            d_vec = [item.split("/")[0] for item in list_of_things]
            i_vec = [int(item.split("/")[1]) for item in list_of_things]
        else:
            raise ValueError(f"Unrecognized type for item0: {type(item0)}, should be int, dict or str")

        if isinstance(item0,(dict,str)):
            list_of_things2 = []
            for d, i in zip(d_vec,i_vec):
                if d!=self.dataset_name:
                    raise NotImplementedError(f"Only {self.dataset_name} is supported. Got dataset_name={d}")
                match_idx = None
                for k,item in enumerate(self.items):
                    if item["i"]==i:
                        match_idx = k
                        break   
                assert match_idx is not None, "No match for dataset_name: "+d+", i: "+str(i)+"Len of dataset: "+str(len(self))
                list_of_things2.append(match_idx)

        assert all([isinstance(item,int) for item in list_of_things2]), "all items in list_of_things must be integers"
        assert all([0<=item<len(self) for item in list_of_things2]), "all items in list_of_things must be valid indices"
        return list_of_things2


    def get_prioritized_sampler(self,pri_didx,seed=None,use_p=True,shuffle=False):
        """
        Returns a sampler which first samples from the dataset 
        with index in pri_idx and then the rest of the dataset
        """
        pri_idx = self.convert_to_idx(pri_didx)
        non_pri_idx = [i for i in range(len(self)) if i not in pri_idx]
        if shuffle:
            if use_p:
                p = self.dataset_weights
            else:
                p = np.ones(len(self))
            gen = torch.Generator().manual_seed(seed)
            sampler = torch.utils.data.WeightedRandomSampler(p,num_samples=len(self),replacement=False,generator=gen)
            new_pri_idx = []
            new_non_pri_idx = []
            for idx in sampler:
                if idx in pri_idx:
                    new_pri_idx.append(idx)
                else:
                    new_non_pri_idx.append(idx)
            pri_idx = new_pri_idx
            non_pri_idx = new_non_pri_idx
        order = pri_idx+non_pri_idx
        return order

    def map_label_to_valid_bits(self,label,info):
        """Map instance IDs in a label image to valid agnostic class IDs."""
        mnc = self.max_num_classes

        valid_mask = label != self.padding_idx
        if np.any(valid_mask):
            uq_labels, uq_counts = np.unique(label[valid_mask], return_counts=True)
        else:
            uq_labels, uq_counts = np.array([], dtype=int), np.array([], dtype=int)

        # If too many classes are present, collapse excess classes before LAP.
        if len(uq_labels) > mnc:
            order = np.argsort(uq_counts)[::-1]
            keep_labels = uq_labels[order[:mnc]]
            excess_labels = uq_labels[order[mnc:]]

            if self.map_excess_classes_to == "largest":
                target = int(keep_labels[0])
                for ex in excess_labels:
                    label[label == int(ex)] = target
            elif self.map_excess_classes_to == "random_different":
                keep_vec = keep_labels.tolist()
                for ex in excess_labels.tolist():
                    candidates = [k for k in keep_vec if k != ex]
                    if len(candidates) == 0:
                        candidates = keep_vec
                    label[label == int(ex)] = int(np.random.choice(candidates))
            elif self.map_excess_classes_to == "random_same":
                target = int(np.random.choice(keep_labels))
                for ex in excess_labels:
                    label[label == int(ex)] = target
            elif self.map_excess_classes_to == "zero":
                target = 0 if 0 in uq_labels else int(keep_labels[0])
                for ex in excess_labels:
                    label[label == int(ex)] = target
            elif self.map_excess_classes_to == "same":
                raise ValueError(
                    f"Found {len(uq_labels)} classes with max_num_classes={mnc} and map_excess_classes_to='same'. "
                    "Please choose a remapping mode such as 'largest'."
                )
            elif self.map_excess_classes_to == "nearest_expensive":
                raise NotImplementedError("nearest_expensive not implemented")
            else:
                raise ValueError(f"Unknown map_excess_classes_to mode: {self.map_excess_classes_to}")

        idx_old = np.unique(label[label!=self.padding_idx].flatten()).tolist()
        idx_old.sort()
        if self.shuffle_labels:
            if not self.shuffle_zero:
                idx_new = np.random.choice(mnc,size=len(idx_old),replace=False).tolist()
            else:
                idx_new = [0]+list(np.random.choice(mnc-1,size=len(idx_old)-1,replace=False)+1)
        else:
            idx_new = idx_old

        old_to_new = {k: v for k,v in zip(idx_old,idx_new)}
        old_to_new[-1] = self.padding_idx

        label = np.vectorize(old_to_new.get)(label)
        info["old_to_new"] = old_to_new
        if self.lap_mode!="none":
            lap_map = self.LAP.mapping_from_labels(label)
            label = np.vectorize(lap_map.get)(label)
            old_to_new_to_lap_map = {k: lap_map.get(v,v) for k,v in old_to_new.items()}
            info["old_to_new"] = old_to_new_to_lap_map
        return label,info
        
    def preprocess(self,image,label,info):
        image,label = self.augment(image,label,info)
        label = label.astype(int)
        augmented = self.sam_aug_small(image=image,mask=label)
        image,label = augmented["image"],augmented["mask"]
        return image,label

    def augment(self,image,label,item):
        augmented = self.augment_fn(image=image,mask=label)
        return augmented["image"],augmented["mask"]
    
    def load_raw_image_label(self,x,longest_side_resize=0,data_root=None):
        if data_root is None:
            data_root = self.data_root
        if isinstance(x,str):
            x = {"dataset_name": x.split("/")[0], "i": int(x.split("/")[1])}
        if isinstance(x,int):
            image_path = os.path.join(data_root,self.items[x]["dataset_name"],self.items[x]["image_path"])
            label_path = os.path.join(data_root,self.items[x]["dataset_name"],self.items[x]["label_path"])
        elif isinstance(x,dict):
            x = copy.deepcopy(x)
            if "image_path" not in x:
                x["image_path"] = load_from_dataset_and_idx(x["dataset_name"],x["i"],im=True)
            if "label_path" not in x:
                x["label_path"] = load_from_dataset_and_idx(x["dataset_name"],x["i"],im=False)
            image_path = os.path.join(data_root,x["dataset_name"],x["image_path"])
            label_path = os.path.join(data_root,x["dataset_name"],x["label_path"])
        else:
            assert isinstance(x,list)
            assert len(x)==2
            image_path,label_path = x
        image = open_image_fast(image_path,num_channels=3)
        label = open_image_fast(label_path,num_channels=1)
        if longest_side_resize>0:
            image = A.LongestMaxSize(max_size=longest_side_resize, interpolation=cv2.INTER_AREA,p=1)(image=image)["image"]
            label = A.LongestMaxSize(max_size=longest_side_resize, interpolation=cv2.INTER_NEAREST, p=1)(image=label)["image"]
        return image,label

    def process_input(self,idx):
        if isinstance(idx,int):
            idx_d = {"idx": idx}
        elif isinstance(idx,str):
            idx_d = {"idx": self.didx_to_item_idx[idx]}
        else:
            assert isinstance(idx,dict), "idx must be an integer or a dictionary or a str, got: "+str(type(idx))
            idx_d = idx
            if "idx" in idx_d:
                pass
            elif "didx" in idx_d:
                idx_d["idx"] = self.didx_to_item_idx[idx["didx"]]
            else:
                assert "i" in idx and "dataset_name" in idx, "idx must contain 'i' and 'dataset_name' keys, or be a didx str or have the 'didx' field"
                idx["idx"] = self.didx_to_item_idx[f"{idx['dataset_name']}/{idx['i']}"]
        return idx_d["idx"]

    def images_to_torch(self,image,label,info):
        image = torch.tensor(image).permute(2,0,1)
        label = torch.tensor(label).unsqueeze(0)
        return image,label,info
    
    def __getitem__(self, idx):
        idx = self.process_input(idx)
        info = copy.deepcopy(self.items[idx])
        dataset_name = info["dataset_name"]
        image_path = os.path.join(self.data_root,dataset_name,info["image_path"])
        label_path = os.path.join(self.data_root,dataset_name,info["label_path"])
        image = open_image_fast(image_path,num_channels=3)
        label = open_image_fast(label_path,num_channels=0) #num_channels=0 means 2D
        image,label = self.preprocess(image,label,info)
        label,info = self.map_label_to_valid_bits(label,info)
        image,label,info = self.images_to_torch(image,label,info)
        info["image"] = image
        return label,info

def load_from_dataset_and_idx(dataset_name,i,im=True):
    if dataset_name != "entity":
        raise NotImplementedError(f"Only entity is supported, got dataset_name={dataset_name}")
    root_dir = Path(__file__).parent.parent.parent / "data" / dataset_name
    train_json = load_json_to_dict_list(str(root_dir / "entityseg_train_lr.json"))
    val_json = load_json_to_dict_list(str(root_dir / "entityseg_val_lr.json"))
    all_images = sorted([im_d["file_name"] for im_d in (train_json["images"] + val_json["images"])])
    if i < 0 or i >= len(all_images):
        raise ValueError(f"Invalid entity index i={i}, valid range is [0, {len(all_images)-1}]")
    fn = all_images[i]
    p = Path(fn)
    if im:
        return fn
    return str(Path(p.parts[0] + "_masks") / f"{p.stem}_mask.png")

def load_raw_image_label(x,longest_side_resize=0,data_root=None):
    if data_root is None:
        data_root = str(Path(__file__).parent.parent.parent / "data")
    return EntityDataset.load_raw_image_label(None,x,longest_side_resize,data_root)

def get_sam_aug(size,padval=-1,imagenet_norm_p=1):
    #SAM uses the default imagenet mean and std, also default in A.Normalize
    sam_aug = A.Compose([A.LongestMaxSize(max_size=size, interpolation=cv2.INTER_AREA, p=1),
                     A.Normalize(p=imagenet_norm_p),
                     A.PadIfNeeded(min_height=size, 
                                   min_width=size, 
                                   border_mode=cv2.BORDER_CONSTANT,
                                   p=1, 
                                   position="top_left")])
    return sam_aug

def open_image_fast(image_path,
                    num_channels=None):
    assert image_path.find(".")>=0, "image_path must contain a file extension"
    extension = image_path.split(".")[-1]
    if extension in ["jpg","jpeg"]:
        image = np.array(Image.open(image_path).convert("RGB"))
    else:
        image = np.array(Image.open(image_path))
    if num_channels is not None:
        assert num_channels in [0,1,3,4], f"Expected num_channels to be in [0,1,3,4], got {num_channels}"
        if num_channels==0: #means only 2 dims
            if (len(image.shape)==3 and image.shape[2]==1):
                image = image[:,:,0]
            else:
                assert len(image.shape)==2, f"loaded image must either be 2D or have 1 channel when num_channels=0. got shape: {image.shape}"
        else:
            if len(image.shape)==2:
                image = image[:,:,None]
            if num_channels==1:
                assert image.shape[2]==1, f"loaded image must have at most 1 channel if num_channels==1, found {image.shape[2]}"
            elif num_channels==3:
                if image.shape[2]==1:
                    image = np.repeat(image,num_channels,axis=-1)
                elif image.shape[2]==4:
                    image = image[:,:,:3]
                else:
                    assert image.shape[2]==3, f"loaded image must have 1,3 or 4 channels if num_channels==3, found {image.shape[2]}"
            elif num_channels==4:
                if image.shape[2]==1:
                    image = np.concatenate([image,image,image,np.ones_like(image)*255],axis=-1)
                elif image.shape[2]==3:
                    image = np.concatenate([image,np.ones_like(image[:,:,0:1])*255],axis=-1)
                else:
                    assert image.shape[2]==4, f"loaded image must have 1,3 or 4 channels if num_channels==4, found {image.shape[2]}"
    return image

def get_augmentation(augment_name="none",s=128,train=True,global_p=1.0,geo_aug_p=0.3):
    list_of_augs = []
    if geo_aug_p>0:
        geo_augs =  [A.Affine(
                        scale=1.0,
                        translate_percent=0.0,
                        rotate=(-20, 20),
                        p=global_p * geo_aug_p
                    )]
    else:
        geo_augs = []
    horz_sym_aug = [A.HorizontalFlip(p=global_p*0.5)]
    all_sym_aug = [A.RandomRotate90(p=global_p*0.5),
                   A.HorizontalFlip(p=global_p*0.5),
                   A.VerticalFlip(p=global_p*0.5)]

    common_color_augs = [A.HueSaturationValue(p=global_p*0.3),
                        A.RGBShift(p=global_p*0.1)]

    common_augs = [A.RandomGamma(p=global_p*0.3),
                A.RandomBrightnessContrast(p=global_p*0.3)]
    
    if augment_name == "none":
        pass
    elif augment_name == "pictures":
        if train:
            list_of_augs.extend(horz_sym_aug)
            list_of_augs.extend(common_augs)
            list_of_augs.extend(common_color_augs)
            list_of_augs.extend(geo_augs)
    elif augment_name == "medical_color":
        if train:
            list_of_augs.extend(all_sym_aug)
            list_of_augs.extend(common_augs)
            list_of_augs.extend(common_color_augs)
            list_of_augs.extend(geo_augs)
    elif augment_name in ["medical_gray","medical_grey"]:
        if train:
            list_of_augs.extend(all_sym_aug)
            list_of_augs.extend(common_augs)
            list_of_augs.extend(geo_augs)
    else:
        raise ValueError("invalid augment_name. Expected one of ['none','pictures','medical_color','medical_gray'] got "+str(augment_name))

    return A.Compose(list_of_augs)

def get_dataset_from_args(args_or_model_id=None,
                          split="vali",
                          prioritized_didx=None,
                          mode="training",
                          return_type="dli",
                          aug_override=None,
                          ):
    if prioritized_didx is not None:
        assert mode=="pri_didx", "mode must be pri_didx if prioritized_didx is provided"
    if args_or_model_id is None:
        #use default args with data
        args_or_model_id = TieredParser().get_args(alt_parse_args=["--model_name","analog_bits"])
        args_or_model_id.datasets = "entity"
    assert split in ["train","vali","test","all",0,1,2,3], f"split must be in ['train','vali','test','all'] or its index, got {split}"
    split = {0:"train",1:"vali",2:"test",3:"all"}.get(split,split)
    assert return_type in ["dli","dl","ds"], f"return_type must be in ['dli','dl','ds'], got {return_type}"
    assert mode in ["training","pure_gen","pri_didx",None], f"sampler_mode must be in ['training','pure_gen','pri_didx',None], got {mode}"
    if isinstance(args_or_model_id,dict):
        args = Namespace(**args_or_model_id)
    elif isinstance(args_or_model_id,str):
        args = load_existing_args(args_or_model_id)
    else:
        assert isinstance(args_or_model_id,Namespace)
        args = args_or_model_id

    if args.encoding_type=="RGB":
        if args.lap_mode=="none":
            max_num_classes = 125
        else:
            max_num_classes = 121
    else:
        max_num_classes = 2**args.diff_channels
    ds = EntityDataset(split=split,
                            image_size=args.image_size,
                            datasets=args.datasets,
                            min_rel_class_area=args.min_label_size,
                            max_num_classes=max_num_classes,
                            shuffle_zero=args.shuffle_zero,
                            padding_idx=-1 if args.ignore_padded else 0,
                            shuffle_labels=args.lap_mode=="none",
                            imagenet_norm=args.imagenet_norm,
                            aug_override=aug_override,
                            global_p=args.aug_prob_multiplier,
                            lap_mode=args.lap_mode,
                            is_rgb=args.encoding_type=="RGB",
                            lap_seed=args.seed,
                            )
    if return_type=="ds":
        return ds
    tbs = args.train_batch_size
    vbs = args.vali_batch_size if args.vali_batch_size>0 else tbs
    bs = {"train": tbs,
          "vali": vbs,
          "test": vbs,
          "all": vbs}[split]
    if mode=="training":
        sampler = ds.get_sampler(args.seed)
    elif mode=="pure_gen":
        sampler = ds.get_gen_dataset_sampler(args.datasets,args.seed)
    elif mode=="pri_didx":
        assert prioritized_didx is not None, "prioritized_didx must be provided if mode is pri_didx"
        sampler = ds.get_prioritized_sampler(prioritized_didx,seed=args.seed)
    dl = torch.utils.data.DataLoader(ds,
                                    batch_size=bs,
                                    sampler=sampler,
                                    shuffle=(sampler is None),
                                    drop_last=mode=="training",
                                    collate_fn=custom_collate_with_info,
                                    num_workers=args.dl_num_workers)
    if return_type=="dl":
        return dl
    elif return_type=="dli":
        return DataloaderIterator(dl)
