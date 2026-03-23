import io
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import glob
from pathlib import Path
from PIL import Image
from source.utils.argparsing import TieredParser
from source.utils.mixed import (bracket_glob_fix, save_dict_list_to_json, 
                                imagenet_preprocess, 
                                load_json_to_dict_list, wildcard_match,
                                unet_kwarg_to_tensor,model_arg_is_trivial)
from source.models.unet import dynamic_image_keys
import cv2
import pandas as pd
import scipy.ndimage as nd

import warnings
try:
    from sklearn.cluster import KMeans
except:
    warnings.warn("Could not import KMeans from sklearn.cluster")

from skimage.measure import find_contours
from matplotlib.patheffects import withStroke
import matplotlib

large_pallete = [  0,   0,   0,  23, 190, 207, 255, 127,  14, 214,  39,  40, 152,
       251, 152,   0,   0, 142, 148, 103, 189, 220, 220,   0, 140,  86,
        75, 107, 142,  35, 220,  20,  60, 255,   0,   0, 255, 255,  90,
       102, 102, 156,  31, 119, 180,   0,   0,  70, 119,  11,  32, 205,
       255,  50,   0,  80, 100, 250, 170,  30,   0,   0, 230, 244,  35,
       232, 227, 119, 194, 255, 220,  80,  44, 160,  44, 190, 153, 153,
       128,  64, 128,   0,  60, 100]

largest_pallete = [  0,   0,   0]+sum([large_pallete[3:] for _ in range(255*3//len(large_pallete)+2)],[])[:255*3]

large_colors = np.array(large_pallete[3:]).reshape(-1, 3)
largest_colors = np.array(largest_pallete[3:]).reshape(-1, 3)


def render_axis_ticks(image_width=1000,
                      num_uniform_spaced=None,
                      bg_color="white",
                      xtick_kwargs={"labels": np.arange(5)},
                      tick_params={}):
    old_backend = matplotlib.rcParams['backend']
    old_dpi = matplotlib.rcParams['figure.dpi']
    dpi = 100
    if num_uniform_spaced is None:
        num_uniform_spaced = len(xtick_kwargs["labels"])
    n = num_uniform_spaced

    matplotlib.rcParams['figure.dpi'] = dpi
    matplotlib.use('Agg')
    fig = None
    buf = None
    try:        
        fig = plt.figure(figsize=(image_width/dpi, 1e-15), facecolor=bg_color)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_facecolor(bg_color)
        ax.set_frame_on(False)
        ax.tick_params(**tick_params)
        fig.add_axes(ax)
        
        plt.yticks([])
        plt.xlim(0, n)
        x_pos = np.linspace(0.5,n-0.5,n)
        if not "ticks" in xtick_kwargs:
            xtick_kwargs["ticks"] = x_pos[:len(xtick_kwargs["labels"])]
        else:
            if xtick_kwargs["ticks"] is None:
                xtick_kwargs["ticks"] = x_pos[:len(xtick_kwargs["labels"])]
        plt.xticks(**xtick_kwargs)

        # Render directly to memory and avoid leaving temporary files behind.
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        im = np.array(Image.open(buf))
        if not im.shape[1]==image_width:
            #reshape with cv2 linear interpolation
            #warnings.warn("Image width is not as expected, likely due to too large text labels. Reshaping with cv2 linear interpolation.")
            im = cv2.resize(im, (image_width, im.shape[0]), interpolation=cv2.INTER_LINEAR)
    finally:
        if buf is not None:
            buf.close()
        if fig is not None:
            plt.close(fig)
        matplotlib.use(old_backend)
        matplotlib.rcParams['figure.dpi'] = old_dpi
    return im

def get_matplotlib_color(color,num_channels=3):
    return render_axis_ticks(23,bg_color=color,xtick_kwargs={"labels": [" "]}, tick_params={"bottom": False})[12,12,:num_channels]

def add_text_axis_to_image(filename_or_image,
                           save_filename=None,
                           n_horz=None,n_vert=None,
                           top=[],bottom=[],left=[],right=[],
                           bg_color="white",
                           xtick_kwargs={},
                           buffer_pixels=4,
                           add_spaces=True):
    """
    Function to take an image filename or numpy array of an image 
    and add text to the top, bottom, left, and right of the image.

    Parameters
    ----------
    filename_or_image : str
        The filename_or_image of the image to modify.
    save_filename : str, optional
        The filename to save the modified image to. The default
        is None (do not save the image).
    n_horz : int, optional
        The number of horizontal text labels to add. The default
        is None (max(len(top),len(bottom))).
    n_vert : int, optional
        The number of vertical text labels to add. The default
        is None (max(len(left),len(right))).
    top : list, optional
        The list of strings to add to the top of the image. The
        default is [].
    bottom : list, optional
        The list of strings to add to the bottom of the image. The
        default is [].
    left : list, optional
        The list of strings to add to the left of the image. The
        default is [].
    right : list, optional
        The list of strings to add to the right of the image. The
        default is [].
    bg_color : list, optional
        The background color of the text. The default is [1,1,1]
        (white).
    xtick_kwargs : dict, optional
        The keyword arguments to pass to matplotlib.pyplot.xticks.
        The default is {}.        
    buffer_pixels : int, optional
        The number of pixels to add as a buffer between the image
        and the text. The default is 4.
    add_spaces : bool, optional
        If True, then a space is added to the beginning and end of
        each label. The default is True.
        
    Returns
    -------
    im2 : np.ndarray
        The modified image with the text axis added.
    """
    if n_horz is None:
        n_horz = max(len(top),len(bottom))
    if n_vert is None:
        n_vert = max(len(left),len(right))
    if isinstance(filename_or_image,np.ndarray):
        im = filename_or_image
        if not np.uint8==im.dtype:
            im = (im*255).astype(np.uint8)
    else:
        assert os.path.exists(filename_or_image), f"filename {filename_or_image} does not exist"
        im = np.array(Image.open(filename_or_image))
    h,w,c = im.shape
    xtick_kwargs_per_pos = {"top":    {"rotation": 0,  "labels": top},
                            "bottom": {"rotation": 0,  "labels": bottom},
                            "left":   {"rotation": 90, "labels": left},
                            "right":  {"rotation": 90, "labels": right}}
    tick_params_per_pos = {"top":    {"top":True, "labeltop":True, "bottom":False, "labelbottom":False},
                           "bottom": {},
                           "left":   {},
                           "right":  {"top":True, "labeltop":True, "bottom":False, "labelbottom":False}}
    pos_renders = {}
    pos_sizes = {}
    for pos in ["top","bottom","left","right"]:
        if len(xtick_kwargs_per_pos[pos]["labels"])==0:
            pos_renders[pos] = np.zeros((0,0,c),dtype=np.uint8)
            pos_sizes[pos] = 0
            continue
        xk = dict(**xtick_kwargs_per_pos[pos],**xtick_kwargs)
        if add_spaces:
            xk["labels"] = [" "+l+" " for l in xk["labels"]]
            if pos=="bottom":
                xk["labels"] = [l+"\n" for l in xk["labels"]]
        if not "ticks" in xk.keys():
            n = n_horz if pos in ["top","bottom"] else n_vert

            if len(xk["labels"])<n:
                xk["labels"] += [""]*(n-len(xk["labels"]))
            elif len(xk["labels"])>n:
                xk["labels"] = xk["labels"][:n]
            else:
                assert len(xk["labels"])==n
        pos_renders[pos] = render_axis_ticks(image_width=w if pos in ["top","bottom"] else h,
                                             num_uniform_spaced=n,
                                             bg_color=bg_color,
                                             xtick_kwargs=xk,
                                             tick_params=tick_params_per_pos[pos])[:,:,:c]
        pos_sizes[pos] = pos_renders[pos].shape[0]
    bg_color_3d = get_matplotlib_color(bg_color,c)
    bp = buffer_pixels
    im2 = np.zeros((h+pos_sizes["top"]+pos_sizes["bottom"]+bp*2,
                    w+pos_sizes["left"]+pos_sizes["right"]+bp*2,
                    c),dtype=np.uint8)
    im2 += bg_color_3d
    im2[bp+pos_sizes["top"]:bp+pos_sizes["top"]+h,
        bp+pos_sizes["left"]:bp+pos_sizes["left"]+w] = im
    #make sure we have uint8
    pos_renders = {k: np.clip(v,0,255) for k,v in pos_renders.items()}
    for pos in ["top","bottom","left","right"]:
        if pos_renders[pos].size==0:
            continue
        if pos=="top":
            im2[bp:bp+pos_sizes["top"],bp+pos_sizes["left"]:bp+pos_sizes["left"]+w] = pos_renders["top"]
        elif pos=="bottom":
            im2[bp+pos_sizes["top"]+h:-bp,bp+pos_sizes["left"]:bp+pos_sizes["left"]+w] = pos_renders["bottom"]
        elif pos=="left":
            im2[bp+pos_sizes["top"]:bp+pos_sizes["top"]+h,bp:bp+pos_sizes["left"]] = np.rot90(pos_renders["left"],k=3)
        elif pos=="right":
            im2[bp+pos_sizes["top"]:bp+pos_sizes["top"]+h,bp+pos_sizes["left"]+w:-bp] = np.rot90(pos_renders["right"],k=3)
    if save_filename is not None:
        Image.fromarray(im2).save(save_filename)
    return im2

def distance_transform_edt_border(mask):
    padded = np.pad(mask,1,mode="constant",constant_values=0)
    dist = nd.distance_transform_edt(padded)
    return dist[1:-1,1:-1]


def darker_color(x,power=2,mult=0.5):
    assert isinstance(x,np.ndarray), "darker_color expects an np.ndarray"
    is_int_type = x.dtype in [np.uint8,np.uint16,np.int8,np.int16,np.int32,np.int64]
    if is_int_type:
        return np.round(255*darker_color(x/255,power=power,mult=mult)).astype(np.uint8)
    else:
        return np.clip(x**power*mult,0,1)
    
class RenderMatplotlibAxis:
    def __init__(self, height, width=None, with_axis=False, set_lims=False, with_alpha=False, dpi=100, show_im=False):
        self.image_to_show = None
        if (width is None) and isinstance(height, (tuple, list)):
            #height is a shape
            height,width = height[:2]
        elif (width is None) and isinstance(height, np.ndarray):
            if show_im:
                self.image_to_show = height
                if self.image_to_show.dtype in [np.float32,np.float64]:
                    self.image_to_show = (self.image_to_show*255).astype(np.uint8)
            #height is an image
            height,width = height.shape[:2]
        elif width is None:
            width = height
        assert isinstance(height, int), f"expected height to be an int after processing, found {type(height)}"
        assert isinstance(width, int), f"expected width to be an int after processing, found {type(width)}"
        if show_im:
            assert self.image_to_show is not None, "expected the first input (height) to be an image if show_im is True"
        self.with_alpha = with_alpha
        self.width = width
        self.height = height
        self.dpi = dpi
        self.old_backend = matplotlib.rcParams['backend']
        self.old_dpi = matplotlib.rcParams['figure.dpi']
        self.fig = None
        self.ax = None
        self._image = None
        self.with_axis = with_axis
        self.set_lims = set_lims

    @property
    def image(self):
        return self._image[:,:,:(3+int(self.with_alpha))]

    def __enter__(self):
        matplotlib.rcParams['figure.dpi'] = self.dpi
        matplotlib.use('Agg')
        figsize = (self.width/self.dpi, self.height/self.dpi)
        self.fig = plt.figure(figsize=figsize,dpi=self.dpi)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        if not self.with_axis:
            self.ax.set_frame_on(False)
            self.ax.get_xaxis().set_visible(False)
            self.ax.get_yaxis().set_visible(False)
        self.fig.add_axes(self.ax)
        if self.image_to_show is not None:
            self.ax.imshow(self.image_to_show)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            # If no exception occurred, save the image to the _image property
            if self.set_lims:
                self.ax.set_xlim(-0.5, self.width-0.5)
                self.ax.set_ylim(self.height-0.5, -0.5)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', pad_inches=0, dpi=self.dpi)
            buf.seek(0)
            self._image = np.array(Image.open(buf))

        plt.close(self.fig)
        matplotlib.use(self.old_backend)
        matplotlib.rcParams['figure.dpi'] = self.old_dpi


def get_mask(mask_vol,idx,onehot=False,onehot_dim=-1):
    if onehot:
        slice_idx = [slice(None) for _ in range(len(mask_vol.shape))]
        slice_idx[onehot_dim] = idx
        return np.expand_dims(mask_vol[tuple(slice_idx)],onehot_dim)
    else:
        return (mask_vol==idx).astype(float)
    
def mask_overlay_smooth(image,
                        mask,
                        num_spatial_dims=2,
                        pallete=None,
                        pixel_mult=1,
                        class_names=None,
                        show_border=False,
                        border_color="darker",
                        alpha_mask=0.4,
                        dont_show_idx=[255],
                        fontsize=12,
                        text_color="class",
                        text_alpha=1.0,
                        text_border_instead_of_background=True,
                        set_lims=True):
    assert isinstance(image,np.ndarray)
    assert isinstance(mask,np.ndarray)
    assert len(image.shape)>=num_spatial_dims, f"image must have at least num_spatial_dims dimensions. Found {len(image.shape)}"
    assert len(mask.shape)>=num_spatial_dims, f"mask must have at least num_spatial_dims dimensions. Found {len(mask.shape)}"
    assert image.shape[:num_spatial_dims]==mask.shape[:num_spatial_dims], f"image shape:{image.shape}\nmask shape: {mask.shape}"
    if pallete is None:
        pallete = np.concatenate([np.array([[0,0,0]]),largest_colors],axis=0)
    if image.dtype==np.uint8:
        was_uint8 = True
        image = image.astype(float)/255
    else:
        was_uint8 = False
    if len(mask.shape)==num_spatial_dims:
        onehot = False
        n = mask.max()+1
        uq = np.unique(mask).tolist()
        mask = np.expand_dims(mask,-1)
    else:
        assert len(mask.shape)==num_spatial_dims+1, "mask must have num_spatial_dims (with integers as classes) or num_spatial_dims+1 dimensions (with onehot encoding)"
        if mask.shape[num_spatial_dims]==1:
            onehot = False
            n = mask.max()+1
            uq = np.unique(mask).tolist()
        else:
            onehot = True
            n = mask.shape[num_spatial_dims]
            uq = np.arange(n).tolist()
    image_colored = image.copy()
    if len(image_colored.shape)==num_spatial_dims:
        image_colored = np.expand_dims(image_colored,-1)
    #make rgb
    if image_colored.shape[-1]==1:
        image_colored = np.repeat(image_colored,3,axis=-1)
    color_shape = tuple([1 for _ in range(num_spatial_dims)])+(3,)
    show_idx = [i for i in uq if (not i in dont_show_idx)]
    for i in show_idx:
        reshaped_color = pallete[i].reshape(color_shape)/255
        mask_coef = alpha_mask*get_mask(mask,i,onehot=onehot)
        image_coef = 1-mask_coef
        #prevent overflow version
        #image_colored = image_colored*image_coef+reshaped_color*mask_coef
        image_colored = (image_colored.astype(float)*image_coef+reshaped_color.astype(float)*mask_coef)
    if class_names is not None:
        assert isinstance(class_names,dict), "class_names must be a dictionary that maps class indices to class names"
        for i in uq:
            assert i in class_names.keys(), f"class_names must have a key for each class index, found i={i} not in class_names.keys()"
    assert isinstance(pixel_mult,int), "pixel_mult must be an integer"
    
    if pixel_mult>1:
        image_colored = cv2.resize(image_colored,None,fx=pixel_mult,fy=pixel_mult,interpolation=cv2.INTER_NEAREST)
    
    image_colored = np.clip(image_colored,0,1)
    if show_border or (class_names is not None):
        image_colored = (image_colored*255).astype(np.uint8)
        h,w = image_colored.shape[:2]
        with RenderMatplotlibAxis(h,w,set_lims=set_lims) as ax:
            plt.imshow(image_colored)
            for i in show_idx:
                mask_coef = get_mask(mask,i,onehot=onehot)
                if pixel_mult>1:
                    mask_coef = cv2.resize(mask_coef,None,fx=pixel_mult,fy=pixel_mult,interpolation=cv2.INTER_LANCZOS4)
                else:
                    mask_coef = mask_coef.reshape(h,w)
                if show_border:                    
                    curves = find_contours(mask_coef, 0.5)
                    if border_color=="darker":
                        border_color_i = darker_color(pallete[i]/255)
                    else:
                        border_color_i = border_color
                    k = 0
                    for curve in curves:
                        plt.plot(curve[:, 1], curve[:, 0], linewidth=1, color=border_color_i)
                        k += 1

                if class_names is not None:
                    t = class_names[i]
                    if len(t)>0:
                        dist = distance_transform_edt_border(mask_coef)
                        y,x = np.unravel_index(np.argmax(dist),dist.shape)
                        if text_color=="class":
                            text_color_i = pallete[i]/255
                        else:
                            text_color_i = text_color
                        text_kwargs = {"fontsize": int(fontsize*pixel_mult),
                                       "color": text_color_i,
                                       "alpha": text_alpha}
                        col_bg = "black" if np.mean(text_color_i)>0.5 else "white"             
                        t = plt.text(x,y,t,**text_kwargs)
                        if text_border_instead_of_background:
                            t.set_path_effects([withStroke(linewidth=3, foreground=col_bg)])
                        else:
                            t.set_bbox(dict(facecolor=col_bg, alpha=text_alpha, linewidth=0))
        image_colored = ax.image
    else:
        if was_uint8: 
            image_colored = (image_colored*255).astype(np.uint8)
    return image_colored

large_pallete = [  0,   0,   0,  23, 190, 207, 255, 127,  14, 214,  39,  40, 152,
       251, 152,   0,   0, 142, 148, 103, 189, 220, 220,   0, 140,  86,
        75, 107, 142,  35, 220,  20,  60, 255,   0,   0, 255, 255,  90,
       102, 102, 156,  31, 119, 180,   0,   0,  70, 119,  11,  32, 205,
       255,  50,   0,  80, 100, 250, 170,  30,   0,   0, 230, 244,  35,
       232, 227, 119, 194, 255, 220,  80,  44, 160,  44, 190, 153, 153,
       128,  64, 128,   0,  60, 100]
large_colors = np.array(large_pallete[3:]).reshape(-1, 3)

def collect_gen_table(gen_id_patterns="all_ade20k[ts_sweep]*",
                   model_id_patterns="*",
                   save=False,
                   return_table=True,
                   save_name="",
                   verbose=True,
                   make_pretty_table=True,
                   pretty_digit_limit=5,
                   search_gen_setups_instead=False,
                   include_mode="last",
                   record_from_sample_opts=[],
                   record_from_args=[],
                   sort_by_key=["save_path"],
                   do_map_to_float=True,
                   round_digits=3,
                   remove_duplicates=True):
    if isinstance(record_from_sample_opts,str):
        record_from_sample_opts = [record_from_sample_opts]
    if isinstance(record_from_args,str):
        record_from_args = [record_from_args]
    assert include_mode in ["last","last_per_gen_id","all"], f"expected include_mode to be one of ['last','last_per_gen_id','all'], found {include_mode}"
    if isinstance(gen_id_patterns,str):
        gen_id_patterns = [gen_id_patterns]
    if isinstance(model_id_patterns,str):
        model_id_patterns = [model_id_patterns]
    model_id_dict = TieredParser().load_and_format_id_dict()
    gen_id_dict = TieredParser("sample_opts").load_and_format_id_dict()
    save_paths = []
    table = pd.DataFrame()
    for model_id,v in model_id_dict.items():
        matched = False
        for model_id_pattern in model_id_patterns:
            if wildcard_match(model_id_pattern,model_id):
                if verbose: 
                    print(f"Matched pattern {model_id_pattern} with model_id {model_id}")
                matched = True
                break
        if matched:
            fn = Path(v["save_path"])/"logging_gen.csv"
            if fn.exists():
                with open(str(fn),"r") as f:
                    column_names = f.readline()[:-1].split(",")
                data = np.genfromtxt(str(fn), dtype=str, delimiter=",")[1:]
                if data.size==0:
                    continue
                if search_gen_setups_instead:
                    file_gen_ids = data[:,column_names.index("gen_setup")].astype(str)
                else:
                    file_gen_ids = data[:,column_names.index("gen_id")].astype(str)
                match_idx = set()
                
                for idx,fgi in enumerate(file_gen_ids):
                    for gen_id_pattern in gen_id_patterns:
                        if wildcard_match(gen_id_pattern,fgi):
                            if verbose: 
                                print(f"Matched pattern {gen_id_pattern} with gen_id {fgi} from model_id {model_id}")
                            match_idx.add(idx)
                            break
                if len(match_idx)==0:
                    continue
                if include_mode=="last":
                    match_idx = [max(match_idx)]
                    if verbose and len(match_idx)>1:
                        print(f"Warning: multiple matches found for model_id {model_id} and gen_ids {data[match_idx,column_names.index('gen_id')]}")
                elif include_mode=="all":
                    match_idx = list(match_idx)
                elif include_mode=="last_per_gen_id":
                    len_before = len(match_idx)
                    match_idx = list(match_idx)
                    match_idx = [max([i for i in match_idx if file_gen_ids[i]==file_gen_ids[j]]) for j in match_idx]
                    if verbose and len(match_idx)<len_before:
                        print(f"Warning: multiple matches found for model_id {model_id} and gen_ids {data[match_idx,column_names.index('gen_id')]}")
                else:
                    match_idx = list(match_idx)
                match_data_s = data[match_idx]
                if len(record_from_args)>0:
                    args = load_json_to_dict_list(str(Path(v["save_path"])/"args.json"))
                    for rfa in record_from_args:
                        assert rfa in args[0].keys(), f"expected record_from_args to be in args, found {rfa}"
                        match_data_s = np.concatenate([match_data_s,np.array([args[0][rfa] for _ in range(match_data_s.shape[0])]).reshape(-1,1)],axis=1)
                        column_names.append(rfa)
                if len(record_from_sample_opts)>0:
                    column_names += record_from_sample_opts
                    empty_array = np.array(["" for _ in range(match_data_s.shape[0])]).reshape(-1,1).repeat(len(record_from_sample_opts),axis=1)
                    match_data_s = np.concatenate([match_data_s,empty_array],axis=1)
                    gen_id_list = match_data_s[:,column_names.index("gen_id")].tolist()
                    for mds_i,gen_id in enumerate(gen_id_list):
                        sample_opts = gen_id_dict[gen_id]
                        for rfso in record_from_sample_opts:
                            match_data_s[mds_i,column_names.index(rfso)] = sample_opts[rfso]
                table = pd.concat([table,pd.DataFrame(match_data_s,columns=column_names)],axis=0)
                save_paths.extend([v["save_path"] for _ in range(len(match_idx))])
            else:
                pass#warnings.warn(f"Could not find file {fn}")
    if table.shape[0]==0:
        warnings.warn("Gen table is empty")
        if return_table:
            return table
        else:
            return
    else:
        if do_map_to_float:
            table = table.map(map_to_float)
        if round_digits>0:
            table = table.round(round_digits)
    table["save_path"] = save_paths
    if isinstance(sort_by_key,str):
        sort_by_key = sort_by_key.split(",")
    table = table.sort_values(by=sort_by_key)
    table = table.loc[:, (table != "").any(axis=0)]
    table_pd = table.copy()
    table = {k: table[k].tolist() for k in table.keys()}
    if make_pretty_table:
        buffer = 2
        pretty_table = ["" for _ in range(len(table["save_path"])+2)] 
        for k in table.keys():
            pretty_col = ["" for _ in range(len(table["save_path"])+2)]
            
            if (isinstance(table[k][0],str)
                and table[k][0].replace(".","").isdigit() 
                and table[k][0].find(".")>=0):
                idx = slice(pretty_digit_limit+2)
            else:
                idx = slice(None)
            max_length = max(max([len(str(x)[idx]) for x in table[k]]),len(k))+buffer
            pretty_col[0] = k+" "*(max_length-len(k))
            pretty_col[1] = "#"*max_length
            pretty_col[2:] = [str(x)[idx]+" "*(max_length-len(str(x)[idx])-2)+", " for x in table[k]]
            if k=="model_name":
                pretty_col[0] = "model_name"+" "*(max_length-len("model_name")-1)+"# "
                pretty_col[1] = "#"*(max_length+1)
                pretty_col[2:] = [s.replace(","," #") for s in pretty_col[2:]]
                pretty_table = [pretty_col[i]+pretty_table[i] for i in range(len(pretty_table))]
            else:
                pretty_table = [pretty_table[i]+pretty_col[i] for i in range(len(pretty_table))]
        table["pretty_table"] = pretty_table
    if save:
        save_dict_list_to_json(table,save_name,append=True)
    if return_table:
        return table_pd

def map_to_float(x):
    try:
        return float(x)
    except:
        return x

def get_dtype(vec):
    vec0 = vec[0]
    assert isinstance(vec0,(str,int,float,bytes)), f"expected vec0 to be a str, int, float, or bytes, found {type(vec0)}"
    try:
        int(vec0)
        return int
    except:
        pass
    try:
        float(vec0)
        return float
    except:
        pass
    return str

def make_loss_plot(save_path,
                   step,
                   save=True,
                   show=False,
                   fontsize=14,
                   figsize_per_subplot=(8,2),
                   remove_old=True,
                   is_ambiguous=False):
    filename = os.path.join(save_path,"logging.csv")
    filename_gen = os.path.join(save_path,"logging_gen.csv")
    filename_step = os.path.join(save_path,"logging_step.csv")
    filenames = [filename_gen,filename_step,filename]
    #helpers
    #get logging index
    gli = lambda s: [s.startswith(p) for p in ["gen_","step_",""]].index(True)
    #get logging string
    gls = lambda s: ["gen_","step_",""][gli(s)]

    all_logging = {}
    for i in range(len(filenames)):
        fn = filenames[i]
        if not os.path.exists(fn):
            continue
        with open(fn,"r") as f:
            column_names = f.readline()[:-1].split(",")
        data = np.genfromtxt(fn, dtype=object, delimiter=",")[1:]
        data[data==b''] = b'nan'
        if data.size==0:
            continue
        if len(data.shape)==1:
            data = np.expand_dims(data,0)
        #inf_mask = np.logical_and(~np.any(np.isinf(data),axis=1),~np.all(np.isnan(data),axis=1))
        #data = data[inf_mask]
        
        if filename_step==fn:
            column_names.append("step")
            data = np.concatenate([data,np.arange(1,len(data)+1).reshape(-1,1)],axis=1)
        for j,k in enumerate(column_names):
            try:
                all_logging[["gen_","step_",""][i]+k] = data[:,j].astype(get_dtype(data[:,j]))
            except:
                print(f"Data shape {data.shape}, j={j}, k={k}, i={i}, column_names={column_names}")
                
    if len(all_logging.keys())==0:
        return
    plot_columns = [["loss","vali_loss"],
                    ["mse_x","vali_mse_x"],
                    ["mse_eps","vali_mse_eps"],
                    ["iou","vali_iou"],
                    ["gen_GED"] if is_ambiguous else ["gen_hiou_e","gen_max_hiou_e"],
                    ["gen_iou"] if is_ambiguous else ["gen_ari","gen_max_ari"],
                    ["step_loss"],
                    ["likelihood","vali_likelihood"]]
    plot_columns_new = []
    #remove non-existent columns
    for i in range(len(plot_columns)):
        plot_columns_i = []
        for s in plot_columns[i]:
            if s in all_logging.keys():
                plot_columns_i.append(s)
        if len(plot_columns_i)>0:
            plot_columns_new.append(plot_columns_i)
    plot_columns = plot_columns_new
    n = len(plot_columns)
    #at most 4 plots per column
    n1 = min(4,n)
    n2 = int(max(1,np.ceil(n/4)))
    
    if "gen_gen_setup" in all_logging.keys():
        plot_gen_setups = np.unique(all_logging["gen_gen_setup"])
    
    figsize = (figsize_per_subplot[0]*n2,figsize_per_subplot[1]*n1)
    fig = plt.figure(figsize=figsize)
    try:
        for i in range(n):
            plt.subplot(n1,n2,i+1)
            Y = []
            for j in range(len(plot_columns[i])):
                name = plot_columns[i][j]
                y = all_logging[name]
                x = all_logging[gls(name)+"step"]
                nan_or_inf_mask = np.logical_or(np.isnan(y),np.isinf(y))
                if gls(name)=="gen_":
                    for k,gen_setup in enumerate(plot_gen_setups):
                        setup_mask = np.array(all_logging["gen_gen_setup"])==gen_setup
                        #gen_id_mask = np.array(all_logging["gen_gen_id"])==""
                        setup_mask = np.logical_and(setup_mask,~nan_or_inf_mask)
                        y2 = y[setup_mask]
                        x2 = x[setup_mask]
                        if len(y2)>0:
                            Y.append(y2)
                            plot_kwargs = get_plot_kwargs(gen_setup+"_"+name,idx=k,y=y2)
                            plt.plot(x2,y2,**plot_kwargs)
                else:
                    y = y[~nan_or_inf_mask]
                    x = x[~nan_or_inf_mask]
                    if len(y)>0:
                        Y.append(y)
                        plot_kwargs = get_plot_kwargs(name,idx=None,y=y)
                        plt.plot(x,y,**plot_kwargs)
            if len(Y)>0:
                plt.legend()
                plt.grid()
                xmax = x.max()
                plt.xlim(0,xmax)
                Y = np.array(sum([y.flatten().tolist() for y in Y],[]))
                if any(np.isinf(Y)):
                    print("Warning: inf found in Y at plot_columns[i]=",plot_columns[i])
                ymin,ymax = Y.min(),Y.max()
                ymax += 0.1*(ymax-ymin)+1e-14
                if name.find("loss")>=0 or name.find("grad_norm")>=0:
                    plt.yscale("log")
                    if ymin<1e-8:
                        ymin = 1e-8
                else:
                    ymin -= 0.1*(ymax-ymin)
                plt.ylim(ymin,ymax)
                plt.xlim(0,xmax*1.05)
                plt.xlabel("steps")
        plt.tight_layout()
        if show:
            plt.show()
        save_name = os.path.join(save_path, f"loss_plot_{step:06d}.png")
        if save:
            fig.savefig(save_name)
        if remove_old:
            clean_up(save_name)
    finally:
        plt.close(fig)

def get_plot_kwargs(name,idx,y):
    plot_kwargs = {"color": None,
                   "label": name}
    if name.find("gen_")>=0:
        if idx is not None:
            plot_kwargs["color"] = f"C{idx}"
    if name.find("max_")>=0:
        plot_kwargs["linestyle"] = "--"
    if len(y)<=25:
        plot_kwargs["marker"] = "o"
    return plot_kwargs

def distance_transform_edt_border(mask):
    padded = np.pad(mask,1,mode="constant",constant_values=0)
    dist = nd.distance_transform_edt(padded)
    return dist[1:-1,1:-1]

"""def analog_bits_on_image(x_bits,im,ab):
    assert isinstance(x_bits,torch.Tensor), "analog_bits_on_image expects a torch.Tensor"
    x_int = ab.bit2int(x_bits.unsqueeze(0)).cpu().detach().numpy().squeeze(0)
    magnitude = np.minimum(torch.min(x_bits.abs(),0)[0].cpu().detach().numpy(),1)
    mask = np.zeros((im.shape[0],im.shape[1],2**ab.num_bits))
    for i in range(2**ab.num_bits):
        mask[:,:,i] = (x_int==i)*magnitude
    return mask_overlay_smooth(im,mask,alpha_mask=1.0)
"""

def mean_dim0(x):
    assert isinstance(x,torch.Tensor), "mean_dim2 expects a torch.Tensor"
    return (x*0.5+0.5).clamp(0,1).mean(0).cpu().detach().numpy()

def replace_nan_inf(x,replace_nan=0,replace_inf=0):
    if torch.is_tensor(x):
        x = x.clone()
        x[torch.isnan(x)] = replace_nan
        x[torch.isinf(x)] = replace_inf
    elif isinstance(x,np.ndarray):
        x = x.copy()
        x[np.isnan(x)] = replace_nan
        x[np.isinf(x)] = replace_inf
    else:
        raise ValueError(f"expected x to be a torch.Tensor or np.ndarray, found {type(x)}")
    return x

def error_image(x):
    return cm.RdBu(replace_nan_inf(255*mean_dim0(x)).astype(np.uint8))[:,:,:3]

def contains_nontrivial_key_val(key,dictionary,ignore_none=True):
    has_key = key in dictionary.keys()
    if has_key:
        if ignore_none:
            has_key = dictionary[key] is not None
    return has_key

def concat_inter_plots(foldername,concat_filename,num_timesteps,remove_children=True,remove_old=True):
    images = []
    filenames = sorted([str(f) for f in list(Path(foldername).glob("intermediate_*.png"))])
    batch_size = len(filenames)
    for filename in filenames:
        im = np.array(Image.open(filename))
        images.append(im)
    images = np.concatenate(images,axis=0)
    images = Image.fromarray(images)
    images.save(concat_filename)
    left = ["gt_bit","final pred_","image"]*batch_size
    right = ["x_t","pred_bit","pred_eps"]*batch_size
    t_vec = np.array(range(num_timesteps, 0, -1))/num_timesteps
    top = bottom = ["","t="]+[f"{t_vec[j]:.2f}" for j in range(num_timesteps)]
    top[1] = "image"
    _ = add_text_axis_to_image(concat_filename,save_filename=concat_filename,
                           left=left,top=top,right=right,bottom=bottom,xtick_kwargs={"fontsize":20})
    if remove_children:
        for filename in filenames:
            os.remove(filename)
        if len(os.listdir(foldername))==0:
            os.rmdir(foldername)
    if remove_old:
        clean_up(concat_filename)

def normal_image(x,imagenet_stats=True): 
    if imagenet_stats:
        x2 = imagenet_preprocess(x.unsqueeze(0),inv=True)
        return x2.squeeze(0).clamp(0,1).cpu().detach().permute(1,2,0).numpy()
    else:
        return (x*0.5+0.5).clamp(0,1).cpu().detach().permute(1,2,0).numpy()

def plot_inter(foldername,
               sample_output,
               model_kwargs,
               ab,
               save_i_idx=None,plot_text=False,imagenet_stats=True):
    t = sample_output["inter"]["t"]
    num_timesteps = len(t)
    
    if save_i_idx is None:
        batch_size = sample_output["pred_bit"].shape[0]
        save_i_idx = np.arange(batch_size)
    else:
        assert isinstance(save_i_idx,list), f"expected save_i_idx to be a list of ints or bools, found {type(save_i_idx)}"
        assert len(save_i_idx)>0, f"expected save_i_idx to be a list of ints or bools, found {save_i_idx}"
        assert isinstance(save_i_idx[0],(bool,int)), f"expected save_i_idx to be a list of ints or bools, found {type(save_i_idx[0])}"
        batch_size = len(save_i_idx)
        if isinstance(save_i_idx[0],bool):
            save_i_idx = np.arange(batch_size)[save_i_idx]
        batch_size = len(save_i_idx)
    image_size = sample_output["pred_bit"].shape[-1]

    map_dict = get_map_dict(imagenet_stats,ab)
    zero_image = np.zeros((image_size,image_size,3))
    has_classes = contains_nontrivial_key_val("classes",model_kwargs)
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    filenames = []
    num_inter_exists = len(glob.glob(bracket_glob_fix(os.path.join(foldername,"intermediate_*.png"))))
    for i in range(batch_size):
        ii = save_i_idx[i]
        images = [[map_dict["gt_bit"](sample_output["gt_bit"][ii])],
                  [map_dict["pred_bit"](sample_output["pred_bit"][ii])],
                  [map_dict["image"](model_kwargs["image"][ii])] if contains_nontrivial_key_val("image",model_kwargs) else [zero_image]]
        images[0].append(zero_image)
        images[1].append(zero_image)
        images[2].append(zero_image)
        text = [["gt_bit"],["final pred_bit"],["image"]]
        for k_i,k in enumerate(["x_t","pred_bit","pred_eps"]):
            for j in range(num_timesteps):
                if k in sample_output["inter"].keys():
                    images[k_i].append(map_dict[k](sample_output["inter"][k][j][i]))
                    text_j = ("    t=" if j==0 else "")+f"{t[j]:.2f}" if k_i==0 else ""
                    text[k_i].append(text_j)
                else:
                    images[k_i].append(zero_image)
                    text[k_i].append("")
        filename = os.path.join(foldername,f"intermediate_{i+num_inter_exists:03d}.png")
        filenames.append(filename)
        images = sum(images,[])
        text = sum(text,[])
        if not plot_text:
            text = ["" for _ in range(len(text))]
        if has_classes:
            text[num_timesteps+3] = f"class={model_kwargs['classes'][ii].item()}"
        montage_save(save_name=filename,
                        show_fig=False,
                        arr=images,
                        padding=1,
                        n_col=num_timesteps+2,
                        text=text,
                        text_color="red",
                        pixel_mult=max(1,128//image_size),
                        text_size=12)

def get_sample_names_from_info(info,newline=True):
    dataset_names = [d["dataset_name"] for d in info]
    datasets_i = [d["i"] for d in info]
    newline = "\n" if newline else ""
    sample_names = [f"{dataset_names[i]}/{newline}{datasets_i[i]}" for i in range(len(datasets_i))]
    return sample_names

def plot_grid(filename,
              output,
              ab,
              max_images=32,
              remove_old=False,
              measure='ari',
              text_inside=True,
              sample_names=None,
              imagenet_stats=True,
              show_keys=dynamic_image_keys+["image","gt_bit","pred_bit"]):
    if isinstance(sample_names,list):
        sample_names = get_sample_names_from_info(sample_names)
    k0 = "pred_bit"
    assert k0 in output.keys(), f"expected output to have key {k0}, found {output.keys()}"
    bs = len(output[k0])
    image_size = output[k0].shape[-1]
    if bs>max_images:
        bs = max_images
    map_dict = get_map_dict(imagenet_stats,ab)
    for k in list(show_keys):
        if model_arg_is_trivial(output.get(k,None)):
            show_keys.remove(k)
            continue
        if isinstance(output[k],list):
            output[k] = unet_kwarg_to_tensor(output[k])
        assert isinstance(output[k],torch.Tensor), f"expected output[{k}] to be a torch.Tensor, found {type(output[k])}"
        assert output[k].shape[-1]==image_size, f"expected output[{k}].shape[2] to be {image_size}, found {output.shape[2]}"
        assert output[k].shape[-2]==image_size, f"expected output[{k}].shape[1] to be {image_size}, found {output.shape[1]}"
        output[k] = output[k][:bs]
        assert k in map_dict.keys(), f"No plotting method found in map_dict for key {k}"
    has_classes = False
    if "classes" in output.keys():
        if output["classes"] is not None:
            has_classes = True
    num_votes = output[k0].shape[1]
    images = []
    text = []
    for k in show_keys:
        if k in output.keys():
            if k==k0:
                for j in range(num_votes):
                    images.extend([map_dict[k](output[k][i][j]) for i in range(bs)])
                    text1 = [k if text_inside else ""]+[""]*(bs-1)
                    text2 = ([f"\n{output[measure][i][j]*100:.0f}" for i in range(bs)]) if measure in output.keys() else (["" for i in range(bs)])
                    #text2 = ["" for i in range(bs)]
                    if j==0:
                        text2[0] = f"\n{measure}={text2[0][1:]}"
                    text.extend([t1+t2 for t1,t2 in zip(text1,text2)])
            else:
                text.extend([k if text_inside else ""]+[""]*(bs-1))
                images.extend([map_dict[k](output[k][i]) for i in range(bs)])
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    montage_save(save_name=filename,
                    show_fig=False,
                    arr=images,
                    padding=1,
                    n_col=bs,
                    text=text,
                    text_color="red",
                    pixel_mult=max(1,64//image_size),
                    text_size=12)
    if sample_names is None:
        sample_names = ["s#"+str(i) for i in range(bs)]
    else:
        pass
    if not text_inside:
        idx = show_keys.index("pred_bit")
        show_keys2 = show_keys[:idx]+["pred_bit\n#"+str(i) for i in range(num_votes)]+show_keys[idx+1:]
        if has_classes:
            bottom_names = [f"class={output['classes'][i].item()}" for i in range(bs)]
        else:
            bottom_names = ["" for i in range(bs)]
        add_text_axis_to_image(filename,
                            save_filename=filename,
                            top=sample_names,
                            bottom=bottom_names,
                            left=show_keys2,
                            right=show_keys2)
    if remove_old:
        clean_up(filename)

def likelihood_image(x):
    return cm.inferno(replace_nan_inf(255*mean_dim0(x*2-1)).astype(np.uint8))[:,:,:3]

def get_zero_im(x):
    return np.zeros((x.shape[-2],x.shape[-1],3))+0.5

def bit_to_np(x,ab):
    return ab.bit2prob(x.unsqueeze(0))[0].permute(1,2,0).cpu().numpy()

def aboi_memory_efficient(x,ab):
    """
    Memory efficient version of aboi (analog bits on image), 
    which uses less memory by not storing the intermediate results.
    """
    if isinstance(x,torch.Tensor):
        x = x.cpu().detach().numpy()
    assert isinstance(x,np.ndarray), "aboi_memory_efficient expects a numpy array"
    assert len(x.shape)==3, "aboi_memory_efficient expects a 3D numpy array"
    color_im = ab.bit2color(x[None])[0].transpose((1,2,0))
    return color_im

def get_map_dict(imagenet_stats,ab):
    imgn_s = imagenet_stats
    nb = ab.num_bits

    if nb==1 or nb==3:
        aboi = lambda x: normal_image(x,imagenet_stats=False)
    elif nb>8:
        aboi = lambda x: aboi_memory_efficient(x,ab)
    else:
        if ab.RGB:
            aboi = lambda x: x.cpu().detach().permute(1,2,0).numpy()
        else:
            aboi = lambda x: mask_overlay_smooth(get_zero_im(x),bit_to_np(x,ab),alpha_mask=1.0)
            
    aboi_split = lambda x: mask_overlay_smooth(normal_image(x[-3:],imgn_s),bit_to_np(x[:-3],ab),alpha_mask=0.6)
    err_im = lambda x: error_image(x)
    lik_im = lambda x: likelihood_image(x)
    normal_image2 = lambda x: normal_image(x,imagenet_stats=imagenet_stats)
    aboi_keys = "x_t,pred_bit,pred_eps,gt_bit,gt_eps,self_cond".split(",")
    map_dict = {"image": normal_image2,
                "err_x": err_im,
                "err_eps": err_im,
                "likelihood": lik_im,}
    for k in aboi_keys:
        map_dict[k] = aboi
    for k in dynamic_image_keys:
        map_dict[k] = aboi_split
    return map_dict

def plot_forward_pass(filename,
                      output,
                      metrics,
                      ab,
                      max_images=32,
                      remove_old=True,
                      text_inside=False,
                      sort_samples_by_t=True,
                      sample_names=None,
                      imagenet_stats=True,
                      show_keys=["image","gt_bit","pred_bit","err_x","likelihood","pred_eps","gt_eps","x_t",
                                 "self_cond"]+dynamic_image_keys):
    if isinstance(sample_names,list):
        sample_names = get_sample_names_from_info(sample_names)
    k0 = "x_t" #key which determines batch size and image size
    bs = output[k0].shape[0]
    if bs>max_images:
        bs = max_images
    image_size = output[k0].shape[-1]

    map_dict = get_map_dict(imagenet_stats,ab)
    for k in list(show_keys):
        if k in ["err_x","likelihood"]:
            continue
        if model_arg_is_trivial(output.get(k,None)):
            show_keys.remove(k)
            continue
        if isinstance(output[k],list):
            output[k] = unet_kwarg_to_tensor(output[k])
        assert isinstance(output[k],torch.Tensor), f"expected output[{k}] to be a torch.Tensor, found {type(output[k])}"
        assert len(output[k])==bs, f"expected output[{k}].shape[0] to be {bs}, found {output[k].shape[0]}"
        assert output[k].shape[-1]==image_size, f"expected output[{k}].shape[2] to be {image_size}, found {output.shape[2]}"
        assert output[k].shape[-2]==image_size, f"expected output[{k}].shape[1] to be {image_size}, found {output.shape[1]}"
        assert k in map_dict.keys(), f"No plotting method found in map_dict for key {k}"
        output[k] = output[k][:bs]
    mask = (output["loss_mask"].to(output["gt_bit"].device) if "loss_mask" in output.keys() else 1.0)
    output["err_x"] = (output["pred_bit"]-output["gt_bit"])*mask
    if "mse_x" not in metrics.keys():
        metrics["mse_x"] = torch.mean(output["err_x"]**2,dim=[1,2,3]).tolist()
    
    if sort_samples_by_t:
        perm = torch.argsort(output["t"]).tolist()
    else:
        perm = torch.arange(bs).tolist()
    images = []
    for k in show_keys:
        is_not_none = [output[k][i] is not None for i in range(bs)]
        assert all(is_not_none), f"expected output[{k}] to be not None for all samples, found {is_not_none}"
        images.append([map_dict[k](output[k][i]) for i in perm])
    text = sum([[k if text_inside else ""]+[""]*(bs-1) for k in show_keys],[])

    if text_inside:
        if "err_idx" in show_keys:
            err_idx = show_keys.index("err_x")*bs
            for i in perm:
                text[i+err_idx] += f"\nmse={metrics['mse_x'][i]:.3f}"
        if "x_t" in show_keys:
            x_t_idx = show_keys.index("x_t")*bs
            for i in perm:
                text[i+x_t_idx] += f"\nt={output['t'][i].item():.3f}"
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    montage_save(save_name=filename,
                    show_fig=False,
                    arr=images,
                    padding=1,
                    n_col=bs,
                    text=text,
                    text_color="red",
                    pixel_mult=max(1,128//image_size),
                    text_size=12)
    if sample_names is None:
        sample_names = ["s#"+str(i) for i in perm]
    else:
        sample_names = [sample_names[i] for i in perm]
    if not text_inside:
        t_and_mse = [f"t={output['t'][i].item():.3f}\nmse={metrics['mse_x'][i]:.3f}" for i in perm]
        add_text_axis_to_image(filename,
                            save_filename=filename,
                            top=sample_names,
                            bottom=t_and_mse,
                            left=show_keys,
                            right=show_keys)
    if remove_old:
        clean_up(filename)
        
def clean_up(filename,verbose=False):
    """
    Removes all files in the same folder as filename that have 
    the same name and format except for the last part of the name
    seperated by an underscore. For example, if filename is
    "folder_name/loss_plot_000000.png", then this function will
    remove all files in folder_name that have the same name and
    format except for the last part of the name seperated by an
    underscore. For example, "folder_name/loss_plot_000001.png"
    """
    assert "_" in Path(filename).name, f"filename {filename} does not contain an underscore, which is assumed for clean_up."
    safe_filename = Path(filename)
    glob_str = "_".join(safe_filename.name.split("_")[:-1])+"_*"+safe_filename.suffix
    old_filenames = list(safe_filename.parent.glob(bracket_glob_fix(glob_str)))
    for old_filename in old_filenames:
        if old_filename!=safe_filename:
            if verbose:
                print("\nRemoving old file:",old_filename,", based on from safe file: ",safe_filename.parent)
            os.remove(old_filename)

def index_dict_with_bool(d,bool_iterable,keys=[],num_recursions=1,
                         raise_error_on_wrong_bs=True,
                         ignore_weird_values=False,
                         raise_error_on_recursion_overflow=False):
    bool_kwargs = {"raise_error_on_wrong_bs": raise_error_on_wrong_bs,
              "ignore_weird_values": ignore_weird_values,
              "raise_error_on_recursion_overflow": raise_error_on_recursion_overflow}
    assert isinstance(d,dict), "expected d to be a dict"
    for k,v in d.items():
        if isinstance(v,dict):
            if num_recursions>0:
                d[k] = index_dict_with_bool(v,bool_iterable,keys=keys+[k],num_recursions=num_recursions-1,**bool_kwargs)
            elif raise_error_on_recursion_overflow:
                raise ValueError(f"Recursion overflow at key {k}")
        else:
            d[k] = index_w_bool(v,bool_iterable,keys+[k],**bool_kwargs)
    return d

def index_w_bool(item,bool_iterable,keys,raise_error_on_wrong_bs=True,ignore_weird_values=False,raise_error_on_recursion_overflow=None):
    bs = len(bool_iterable)
        
    if item is not None:
        bs2 = len(item)
        if bs2!=bs:
            if raise_error_on_wrong_bs:
                raise ValueError(f"Expected len(item)={bs}, found {bs2}. type(item)={type(item)}. Keys={keys}")
            else:
                item = None
        if torch.is_tensor(item):
            out = torch.stack([item[i] for i in range(bs) if bool_iterable[i]],dim=0)
        elif isinstance(item,np.ndarray):
            out = np.concatenate([item[i][None] for i in range(bs) if bool_iterable[i]],axis=0)
        elif isinstance(item,list):
            out = [item[i] for i in range(len(item)) if bool_iterable[i]]
        else:
            if ignore_weird_values:
                out = item
            else:
                raise ValueError(f"Expected item to be None, torch.Tensor, np.ndarray, or list, found {type(item)}")
    else:
        out = None
    return out

def montage_save(save_name="test.png",
                 save_fig=True,
                 show_fig=True,
                 pixel_mult = 4,
                 **montage_kwargs
                ):
    """ Save a montage of images to a file (optional) and show it (optional)

    Args:
        save_name (str, optional): name of the file to save the image to. Defaults 
            to "test.png".
        save_fig (bool, optional): should the image be saved. Defaults to True.
        show_fig (bool, optional): should the image be shown. Defaults to True.
        pixel_mult (int, optional): Optional multiplier to be used to make pixels 
            consist of pixel_mult^2 pixels. Text shown on the montage can however
            use the lower resolution, making the text independent from how large
            or small the image is (best to use powers of 2 for nearest neighbour
            interpolation). Defaults to 4.
    """
    fig = plt.figure(frameon=False)
    try:
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        montage_kwargs["create_figure"] = False
        montage_kwargs["return_im"] = True
        montage_im = montage(**montage_kwargs)
        fig.set_size_inches(montage_im.shape[1]*pixel_mult/100,montage_im.shape[0]*pixel_mult/100)
        if save_fig:
            fig.savefig(save_name)
        if show_fig:
            plt.show()
    finally:
        plt.close(fig)


def montage(arr,
            maintain_aspect=True,
            reshape=True,
            text=None,
            return_im=False,
            imshow=True,
            reshape_size=None,
            n_col=None,
            n_row=None,
            padding=0,
            padding_color=0,
            rows_first=True,
            figsize_per_pixel=1/100,
            text_color=[1,0,0],
            text_size=10,
            create_figure=True):
    """
    Displays and returns an montage of images from a list or 
    list of lists of images.

    Parameters
    ----------
    arr : list
        A list or list of lists containing images (np.arrays) of shape 
        (d1,d2), (d1,d2,1), (d1,d2,3) or (d1,d2,4). If arr is a list of lists
        then the first list dimensions is vertical and second is horizontal.
        If there is only one list dimensions then the list will be put in an
        appropriate 2D grid of images. The input can also be a 5D or 4D 
        np.array and in this case the first two dimensions are intepreted 
        the same way as if they were a list. Even if the 5th channel dimension
        is size 1 it has to in included in this case.
    maintain_aspect : boolean, optional
        Should image aspect ratios be maintained. Only relevant if 
        reshape=True. The default is True.
    reshape : boolean, optional
        Should images be reshaped to better fit the montage image. The default 
        is True.
    imshow : boolean, optional
        Should plt.imshow() be used inside the function. The default is True.
    reshape_size : array-like, optional
        2 element list or array like variable. Specifies the number of pixels 
        in the first dim (vertical) and second dim (horizontal) per image in
        the resulting concatenated image
        The default is None.
    n_col : int, optional
        Number of columns the montage will contain.
        The default is None.
    n_row : int, optional
        Number of rows the montage will contain.
        The default is None.
    padding : int or [int,int], optional
        Number of added rows/columns of padding to each image. If an int is
        given the same horizontal and vertical padding is used. If a list is
        given then the first index is the number of vertical padding pixels and
        the second index is the number of horizontal padding pixels. 
        The default is None.
    padding_color : float or int
        The color of the used padding. The default is black (0).
    rows_first : bool
        If True and a single list is given as arr then the images will first
        be filled into row 0 and then row 1, etc. Otherwise columns will be
        filled first. The default is True.
    figsize_per_pixel : float
        How large a figure to render if imshow=True, in relation to pixels.
        Defaults to 1/100.
    text_color : matplotlib color-like
        color of text to write on top of images. Defaults to red ([1,0,0]).
    text_size : float or int
        Size of text to write on top of images. Defaults to 10.
    create_figure : bool
        Should plt.figure() be called when imshow is True? Defaults to True.
    Returns
    -------
    im_cat : np.array
        Concatenated montage image.
        
    
    Example
    -------
    montage(np.random.rand(2,3,4,5,3),reshape_size=(40,50))

    """
    if torch.is_tensor(arr):
        assert len(arr.shape)==4, "torch tensor must have at 4 dims, formatted as (n_images,channels,H,W)"
        arr = arr.detach().cpu().clone().permute(0,2,3,1).numpy()
    if isinstance(arr,np.ndarray):
        if len(arr.shape)==4:
            arr = [arr[i] for i in range(arr.shape[0])]
        elif len(arr.shape)==5:
            if not rows_first:
                arr = np.transpose(arr,(1,0,2,3,4))
            n1 = arr.shape[0]
            n2 = arr.shape[1]
            arr = [[arr[i,j] for j in range(arr.shape[1])]
                   for i in range(arr.shape[0])]
        else:
            raise ValueError("Cannot input np.ndarray with less than 4 dims")
    
    if isinstance(arr[0],np.ndarray): #if arr is a list or 4d np.ndarray
        if (n_col is None) and (n_row is None):
            n1 = np.floor(len(arr)**0.5).astype(int)
            n2 = np.ceil(len(arr)/n1).astype(int)
        elif (n_col is None) and (n_row is not None):
            n1 = n_row
            n2 = np.ceil(len(arr)/n1).astype(int)
        elif (n_col is not None) and (n_row is None):
            n2 = n_col
            n1 = np.ceil(len(arr)/n2).astype(int)
        elif (n_col is not None) and (n_row is not None):
            assert n_col*n_row>=len(arr), "number of columns/rows too small for number of images"
            n1 = n_row
            n2 = n_col
        
        if rows_first:
            arr2 = []
            for i in range(n1):
                arr2.append([])
                for j in range(n2):
                    ii = n2*i+j
                    if ii<len(arr):
                        arr2[i].append(arr[ii])
        else:
            arr2 = [[] for _ in range(n1)]
            for j in range(n2):
                for i in range(n1):
                    ii = i+j*n1
                    if ii<len(arr):
                        arr2[i].append(arr[ii])
        arr = arr2
    if n_row is None:
        n1 = len(arr)
    else:
        n1 = n_row
        
    n2_list = [len(arr[i]) for i in range(n1)]
    if n_col is None:
        n2 = max(n2_list)
    else:
        n2 = n_col
        
    idx = []
    for i in range(n1):
        idx.extend([[i,j] for j in range(n2_list[i])])
    n = len(idx)
    idx = np.array(idx)
    
    N = list(range(n))
    I = idx[:,0].tolist()
    J = idx[:,1].tolist()
    
    D1 = np.zeros(n,dtype=int)
    D2 = np.zeros(n,dtype=int)
    aspect = np.zeros(n)
    im = np.zeros((32,32,3))
    channels = 1
    for n,i,j in zip(N,I,J): 
        if arr[i][j] is None:#image is replaced with zeros of the same size as the previous image
            arr[i][j] = np.zeros_like(im)
        else:
            assert isinstance(arr[i][j],np.ndarray), "images in arr must be np.ndarrays (or None for a zero-image)"
        im = arr[i][j]
        
        D1[n] = im.shape[0]
        D2[n] = im.shape[1]
        if len(im.shape)>2:
            channels = max(channels,im.shape[2])
            assert im.shape[2] in [1,3,4]
            assert len(im.shape)<=3
    aspect = D1/D2
    if reshape_size is not None:
        G1 = reshape_size[0]
        G2 = reshape_size[1]
    else:
        if reshape:
            G2 = int(np.ceil(D2.mean()))
            G1 = int(np.round(G2*aspect.mean()))
        else:
            G1 = int(D1.max())
            G2 = int(D2.max())
    if padding is not None:
        if isinstance(padding,int):
            padding = [padding,padding]
    else:
        padding = [0,0]
        
    p1 = padding[0]
    p2 = padding[1]
    G11 = G1+p1*2
    G22 = G2+p2*2
    
    
    im_cat_size = [G11*n1,G22*n2]

    im_cat_size.append(channels)
    im_cat = np.zeros(im_cat_size)
    if channels==4:
        im_cat[:,:,3] = 1

    for n,i,j in zip(N,I,J): 
        im = arr[i][j]
        if issubclass(im.dtype.type, np.integer):
            im = im.astype(float)/255
        if not reshape:
            d1 = D1[n]
            d2 = D2[n]
        else:
            z_d1 = G1/D1[n]
            z_d2 = G2/D2[n]
            if maintain_aspect:
                z = [min(z_d1,z_d2),min(z_d1,z_d2),1][:len(im.shape)]
            else:
                z = [z_d1,z_d2,1][:len(im.shape)]
            im = nd.zoom(im,z)
            d1 = im.shape[0]
            d2 = im.shape[1]
            
        if len(im.shape)==3:
            im = np.pad(im,((p1,p1),(p2,p2),(0,0)),constant_values=padding_color)
        elif len(im.shape)==2:
            im = np.pad(im,((p1,p1),(p2,p2)),constant_values=padding_color)
        else:
            raise ValueError("images in arr must have 2 or 3 dims")
            
        d = (G1-d1)/2
        idx_d1 = slice(int(np.floor(d))+i*G11,G11-int(np.ceil(d))+i*G11)
        d = (G2-d2)/2
        idx_d2 = slice(int(np.floor(d))+j*G22,G22-int(np.ceil(d))+j*G22)
        
        if len(im.shape)>2:
            im_c = im.shape[2]
        else:
            im_c = 1
            im = im[:,:,None]
            
        if im_c<channels:
            if channels>=3 and im_c==1:
                if len(im.shape)>2:
                    im = im[:,:,0]
                im = np.stack([im]*3,axis=2)
            if channels==4 and im_c<4:
                im = np.concatenate([im]+[np.ones((im.shape[0],im.shape[1],1))],axis=2)
        im_cat[idx_d1,idx_d2,:] = im
    #im_cat = np.clip(im_cat,0,1)
    if imshow:
        if create_figure:
            plt.figure(figsize=(figsize_per_pixel*im_cat.shape[1],figsize_per_pixel*im_cat.shape[0]))
        
        is_rgb = channels>=3
        if is_rgb:
            plt.imshow(np.clip(im_cat,0,1),vmin=0,vmax=1)
        else:
            plt.imshow(im_cat,cmap="gray")

        if text is not None:
            #max_text_len = max([max(list(map(len,str(t).split("\n")))) for t in text])
            #text_size = 10#*G22/max_text_len*figsize_per_pixel #42.85714=6*16/224/0.01
            for i,j,t in zip(I,J,text):
                dt1 = p1+G11*i
                dt2 = p2+G22*j
                plt.text(x=dt2,y=dt1,s=str(t),color=text_color,va="top",ha="left",size=text_size)

    if return_im:
        return im_cat