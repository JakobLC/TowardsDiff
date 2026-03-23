import torch
import numpy as np
import matplotlib
from argparse import Namespace
import os
import warnings
import tqdm
from source.models.unet import all_input_keys
from source.utils.mixed import (get_time,save_dict_list_to_json,
                   check_keys_are_same,mask_from_imshape,
                   unet_kwarg_to_tensor,
                   nice_split)
from source.utils.metric_and_loss import get_segment_metrics
from source.utils.dataloading import get_dataset_from_args
from source.utils.plot import plot_grid,plot_inter,concat_inter_plots
from source.utils.argparsing import TieredParser, save_args, overwrite_existing_args, str2bool
from source.utils.analog_bits import AnalogBits
from pathlib import Path
import copy

DEFAULT_NUM_INTER_STEPS = 10
DEFAULT_NUM_INTER_SAMPLES = 8
DEFAULT_NUM_GRID_SAMPLES = 8
DEFAULT_INTER_VOTES_PER_SAMPLE = 1

class DiffusionSampler(object):
    def __init__(self, trainer, opts=None,
                 ):
        super().__init__()
        if opts is None:
            opts = TieredParser("sample_opts").get_args([])
        self.opts = opts
        self.trainer = trainer
        self.args = copy.deepcopy(self.trainer.args)
        self.ab = AnalogBits(self.args)
        self.opts.seed = self.args.seed
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            print("WARNING: CUDA not available. Using CPU.")
            self.device = torch.device("cpu")

    def prepare_sampling(self,model=None):
        #init variables
        self.samples = []
        self.light_stats = []
        self.source_idx = 0
        self.bss = 0
        self.source_batch = None
        self.queue = None
        self.eval_batch_size = self.opts.eval_batch_size if self.opts.eval_batch_size>0 else self.args.train_batch_size
        if self.args.mode=="gen":
            self.args_restore = copy.deepcopy(self.args)
            self.args.dl_num_workers = 0
            if len(self.opts.datasets)>0:
                if not isinstance(self.opts.datasets,list):
                    self.opts.datasets = self.opts.datasets.split(",")
                self.args.datasets = self.opts.datasets
            
            if isinstance(self.opts.aug_override,str):
                if self.opts.aug_override.lower()=="none":
                    aug_override = None
                else:
                    aug_override = str2bool(self.opts.aug_override) 
            else:
                aug_override = str2bool(self.opts.aug_override) 

            self.dataloader = get_dataset_from_args(self.args,
                                                split=self.opts.split,
                                                mode="pure_gen",
                                                aug_override=aug_override)
        else:
            assert hasattr(self.trainer,f"{self.opts.split}_dl"), f"trainer does not have a dataloader for split={self.opts.split}."
            self.dataloader = getattr(self.trainer,f"{self.opts.split}_dl")

        lpd = self.dataloader.dataloader.dataset.len_per_dataset
        datasets = self.args.datasets if isinstance(self.args.datasets,list) else [self.args.datasets]
        max_num_samples = sum([lpd[dataset] for dataset in datasets])
        if self.opts.num_samples<0:
            self.opts.num_samples = max_num_samples
        elif self.opts.num_samples>max_num_samples:
            print(f"WARNING: num_samples={self.opts.num_samples} is larger than the maximum number of samples in the specified datasets: {max_num_samples}. Setting num_samples to the maximum.")
            self.opts.num_samples = max_num_samples

        if model is None:
            if self.opts.ema_idx>=0:
                model, self.swap_pointers_func = self.trainer.get_ema_model(self.opts.ema_idx)
            else:
                model = self.trainer.model
        was_training = model.training
        model.eval()
        #print first 10 params of model

        if self.opts.do_agg:
            old_backend = matplotlib.get_backend()
            matplotlib.use("agg")
        else:
            old_backend = None
        if self.opts.default_save_folder=="":
            self.opts.default_save_folder = os.path.join(self.args.save_path,"samples")
        def_save_name = f"{self.opts.gen_id}_{self.trainer.step:06d}"
        if self.opts.save_light_stats:
            if self.opts.light_stats_filename=="":
                self.opts.light_stats_filename = os.path.join(self.opts.default_save_folder,f"light_stats_{def_save_name}.json")
        if "grid" in self.opts.plotting_functions.split(","):
            if self.opts.grid_filename=="":
                self.opts.grid_filename = os.path.join(self.opts.default_save_folder,f"grid_{def_save_name}.png")
        inter_is_used = (("inter" in self.opts.plotting_functions.split(",")) or ("concat" in self.opts.plotting_functions.split(",")))
        if inter_is_used:
            legal_timesteps = list(range(self.opts.num_timesteps-1, -1, -1))
            idx = np.round(np.linspace(0,len(legal_timesteps)-1,DEFAULT_NUM_INTER_STEPS)).astype(int)
            #remove duplicates without changing order
            idx = [idx[i] for i in range(len(idx)) if (i==0) or (idx[i]!=idx[i-1])]
            self.save_i_steps = [legal_timesteps[i] for i in idx]
            self.inter_folder = os.path.join(self.opts.default_save_folder,f"inter_{def_save_name}")
            if "concat" in self.opts.plotting_functions.split(","):
                if self.opts.concat_inter_filename=="":
                    self.opts.concat_inter_filename = os.path.join(self.opts.default_save_folder,f"concat_{def_save_name}.png")
        else:
            self.save_i_steps = []
            self.inter_folder = ""

        if self.inter_folder!="":
            os.makedirs(self.inter_folder,exist_ok=True)
        if self.opts.concat_inter_filename!="":
            os.makedirs(os.path.dirname(self.opts.concat_inter_filename),exist_ok=True)
        if self.opts.grid_filename!="":
            os.makedirs(os.path.dirname(self.opts.grid_filename),exist_ok=True)

        return model, was_training, old_backend

    def verify_valid_opts(self):
        assert self.opts.num_samples>0, "num_samples must be positive."
        assert self.opts.num_votes>0, "num_votes must be positive."
        
    def sample(self,model=None,**kwargs):
        self.opts = Namespace(**{**vars(self.opts),**kwargs})
        
        print("Sampling with gen_id:",self.opts.gen_id)
        model,was_training,old_backend = self.prepare_sampling(model)
        self.verify_valid_opts()
        
        self.queue = None
        metric_list = []
        votes = []
        num_batches = np.ceil(self.opts.num_samples*self.opts.num_votes/self.eval_batch_size).astype(int)
        if num_batches==0:
            warnings.warn("num_batches==0.")
            return None
        if self.opts.progress_bar:
            progress_bar = tqdm.tqdm(range(num_batches), desc="Batch progress.")
        else:
            progress_bar = range(num_batches)
        with torch.no_grad():
            for _ in progress_bar:
                gt_int, model_kwargs, info, batch_queue = self.form_next_batch()
                gt_bit = self.ab.int2bit(gt_int)
                x_init = torch.randn_like(gt_bit)
                sample_output = self.trainer.cgd.sample_loop(model=model, 
                                            x_init=x_init, 
                                            num_steps=self.opts.num_timesteps, 
                                            sampler_type=self.opts.sampler_type,
                                            clip_x=self.opts.clip_denoised,
                                            model_kwargs=model_kwargs,
                                            guidance_weight=self.opts.guidance_weight,
                                            progress_bar=self.opts.progress_bar_timestep,
                                            save_i_steps=self.save_i_steps,
                                            save_i_idx=[bq["save_inter_steps"] for bq in batch_queue],
                                            guidance_kwargs=self.opts.guidance_kwargs,
                                            save_entropy=False,
                                            replace_padding=True,
                                            imshape=[info_i["imshape"] for info_i in info],
                                            )
                for k in model_kwargs.keys():
                    model_kwargs[k] = unet_kwarg_to_tensor(model_kwargs[k],key=k)
                self.run_on_single_batch(sample_output,batch_queue,gt_bit,model_kwargs,info)
                for i in range(sample_output["pred_bit"].shape[0]):
                    votes.append(sample_output["pred_bit"][i])
                    if batch_queue[i]["vote"]==self.opts.num_votes-1:
                        model_kwargs_i = {k: 
                                          (model_kwargs[k][i] if model_kwargs[k] is not None else None) 
                                          for k in model_kwargs.keys()}
                        metrics = self.run_on_full_votes(votes,gt_int[i],gt_bit[i],info[i],model_kwargs_i,x_init[i],batch_queue[i])
                        votes = []
                        metric_list.append(metrics)

        sample_output, metric_output = self.get_output_dict(metric_list, self.samples)
        self.run_on_finished(output={**sample_output,**metric_output})

        if old_backend is not None:
            matplotlib.use(old_backend)
        if hasattr(self,"args_restore"):
            self.args = copy.deepcopy(self.args_restore)
        if was_training:
            model.train()
        
        metric_output["gen_setup"] = self.opts.gen_setup
        metric_output["gen_id"] = self.opts.gen_id

        if hasattr(self,"swap_pointers_func"):
            self.swap_pointers_func()
            del self.swap_pointers_func
        return None, metric_output

    def get_output_dict(self, metric_list, samples, info_keys_save=["dataset_name","i","gts_didx","imshape"]):
        model_kwargs_keys = []
        for s in samples:
            for k in s["model_kwargs"].keys():
                if k not in model_kwargs_keys:
                    model_kwargs_keys.append(k)
        for k in model_kwargs_keys:
            for i in range(len(samples)):
                if k not in samples[i]["model_kwargs"].keys():
                    samples[i]["model_kwargs"][k] = None

        sample_output = {}
        metric_output = {k: [m[k] for m in metric_list] for k in metric_list[0].keys()}
        #check for key conflicts
        if samples is not None:
            assert check_keys_are_same(samples)
            for k in samples[0].keys():
                if k=="info":
                    #sample_output[k] = [{sub_k: s[k][sub_k] for sub_k in info_keys_save} for s in samples]
                    sample_output[k] = [{sub_k: s[k][sub_k] for sub_k in s[k].keys() if sub_k in info_keys_save} for s in samples]
                    continue
                if isinstance(samples[0][k],dict):
                    assert check_keys_are_same([s[k] for s in samples]), f"Key conflict for key={k}"
                    for sub_k in samples[0][k].keys():                        
                        sample_output[sub_k] = unet_kwarg_to_tensor([s[k][sub_k] for s in samples],key=sub_k)
                elif torch.is_tensor(samples[0][k]) or isinstance(samples[0][k],list):
                    sample_output[k] = unet_kwarg_to_tensor([s[k] for s in samples],key=k)
        return sample_output, metric_output
            
    def run_on_single_batch(self,sample_output,bq,gt_bit,model_kwargs,info):
        sample_output = copy.deepcopy(sample_output)
        model_kwargs = copy.deepcopy(model_kwargs)
        info = copy.deepcopy(info)
        sample_output["gt_bit"] = gt_bit
        if self.inter_folder!="":
            save_i_idx = [bq_i["save_inter_steps"] for bq_i in bq]
            plot_inter(foldername=self.inter_folder,
                       sample_output=sample_output,
                       model_kwargs=model_kwargs,
                       save_i_idx=save_i_idx,
                       plot_text=self.opts.concat_inter_filename=="",
                       imagenet_stats=self.args.crop_method.startswith("sam"),
                       ab=self.ab)
            
    def run_on_full_votes(self,votes,gt_int,gt_bit,info,model_kwargs,x_init,bqi):
        gt_int = gt_int.cpu()
        gt_bit = gt_bit.cpu()
        votes = torch.stack(votes,dim=0).cpu()
        votes_int = self.ab.bit2int(votes,[info]*votes.shape[0])

        imsize = gt_int.shape[-1]
        mask = torch.from_numpy(mask_from_imshape(info["imshape"],imsize,num_dims=3)).to(self.device)

        metrics = []
        for i in range(len(votes)):
            metrics_i = get_segment_metrics(votes_int[i],gt_int,mask=mask,ignore_zero=False)
            metrics.append(metrics_i)
        metrics = {k: [m[k] for m in metrics] for k in metrics[0].keys()}

        save_sample = bqi["save_grid"]
        if save_sample:
            self.samples.append({"pred_bit": votes,
                                "pred_int": votes_int,
                                "gt_bit": gt_bit,
                                "gt_int": gt_int,
                                "x_init": x_init,
                                "info": copy.deepcopy(info),
                                "model_kwargs": model_kwargs})
        if self.opts.save_light_stats:
            light_stats = {"info": {k: v for k,v in info.items() if k in ["split_idx","i","dataset_name","num_classes","imshape","gts_didx"]},
                           "model_kwargs_abs_sum": {k: 
                                                    (v.abs().sum().item() if torch.is_tensor(v) else 0) 
                                                    for k,v in model_kwargs.items()},
                           "metrics": metrics,
                           "has_raw_sample": False}
            for k,v in model_kwargs.items():
                if not torch.is_tensor(v):
                    import jlc
                    print("model_kwargs for k="+k)
                    jlc.shaprint(v)
                    raise ValueError(f"model_kwargs[{k}] is not a tensor.")

            self.light_stats.append(light_stats)
        return metrics
    
    def run_on_finished(self,output):
        self.opts.model_id = self.args.model_id
        self.opts.time = get_time()
        if self.args.mode!="gen":
            try:
                overwrite_existing_args(self.opts)
            except:
                if not Path(self.opts.default_save_folder).exists():
                    Path(self.opts.default_save_folder).mkdir(parents=False, exist_ok=True)
                save_args(self.opts)
        else:
            save_args(self.opts)
        if "grid" in self.opts.plotting_functions.split(","):
            assert self.opts.grid_filename.endswith(".png"), f"filename: {filename}"
            filename = self.opts.grid_filename
            max_images = min(DEFAULT_NUM_GRID_SAMPLES,len(self.samples))
            plot_grid(filename,output,max_images=max_images,remove_old=True,sample_names=output["info"],
                      imagenet_stats=self.args.crop_method.startswith("sam"),ab=self.ab)
        if "concat" in self.opts.plotting_functions.split(","):
            assert self.opts.concat_inter_filename.endswith(".png"), f"filename: {filename}"
            concat_inter_plots(foldername = self.inter_folder,
                               concat_filename = self.opts.concat_inter_filename,
                               num_timesteps = len(self.save_i_steps),
                               remove_children="inter" not in self.opts.plotting_functions.split(","),
                               remove_old = True)
        if self.opts.save_light_stats:
            save_dict_list_to_json(self.light_stats,self.opts.light_stats_filename)

    def sampler_get_kwargs(self):
        if self.opts.kwargs_mode in ["train","train_image"]:
            assert self.trainer is not None, "self.trainer is None. Set self.trainer to a DiffusionModelTrainer instance or a class with a usable get_kwargs() method."
            gt_int,model_kwargs,info = self.trainer.get_kwargs(next(self.dataloader),
                                                          force_image=self.opts.kwargs_mode=="train_image")
        else:
            if self.args.mode=="gen" or self.args.dl_num_workers==0:
                self.dataloader.dataloader.dataset.gen_mode = True #enables all dynamic cond inputs
            gt_int,model_kwargs,info = self.trainer.get_kwargs(next(self.dataloader), gen=True)
            if self.args.mode=="gen" or self.args.dl_num_workers==0:
                self.dataloader.dataloader.dataset.gen_mode = False

            model_kwargs_use = ["image"]

            do_nothing_kwargs_modes = ["none","only_image","image",""]
            special_kwargs_modes = ["all","train","train_image"]+do_nothing_kwargs_modes
            if self.opts.kwargs_mode in do_nothing_kwargs_modes:
                pass
            elif self.opts.kwargs_mode=="all":
                model_kwargs_use.extend(all_input_keys)
            else:
                for k in nice_split(self.opts.kwargs_mode):
                    if k not in all_input_keys:
                        raise ValueError(f"If kwargs_mode is NOT a special value ({special_kwargs_modes})"
                                         f"then it must be a comma-separated list of valid keys from all_input_keys." 
                                         f"Found: {k}. all_input_keys: {all_input_keys}")
                model_kwargs_use.extend(nice_split(self.opts.kwargs_mode))

            if not all([k in model_kwargs.keys() for k in model_kwargs_use]):
                not_found_kwargs = [k for k in model_kwargs_use if k not in model_kwargs.keys()]
                raise ValueError(f"Could not find the following requested kwargs from the dataloader: {not_found_kwargs}")
            model_kwargs = {k: model_kwargs[k] for k in model_kwargs_use}
        return gt_int,model_kwargs,info
            
    def form_next_batch(self):
        if self.queue is None:
            self.queue = []
            for i in range(self.opts.num_samples):
                for j in range(self.opts.num_votes):
                    save_inter_steps = (i<DEFAULT_NUM_INTER_SAMPLES) and (j<DEFAULT_INTER_VOTES_PER_SAMPLE)
                    self.queue.append({"sample":i,
                                       "vote":j,
                                       "save_inter_steps": save_inter_steps, 
                                       "save_grid": (i<DEFAULT_NUM_GRID_SAMPLES)})
        
        bs = min(self.eval_batch_size,len(self.queue))
        if self.source_idx >= self.bss:
            self.source_batch = self.sampler_get_kwargs()
            self.bss = self.source_batch[0].shape[0]
            self.source_idx = 0
        batch_x = []
        #use_kwargs = [k for k,v in self.source_batch[1].items() if v is not None]
        use_kwargs = list(self.source_batch[1].keys())
        batch_kwargs = {k: [] for k in use_kwargs}
        batch_info = []
        batch_queue = []
        for i in range(bs):
            batch_queue.append(self.queue.pop(0))
            batch_x.append(self.source_batch[0][self.source_idx])
            for k in use_kwargs:
                if self.source_batch[1][k] is None:
                    batch_kwargs[k].append(None)
                else:
                    batch_kwargs[k].append(self.source_batch[1][k][self.source_idx])
            batch_info.append(self.source_batch[2][self.source_idx])
            
            if batch_queue[-1]["vote"]==self.opts.num_votes-1:
                self.source_idx += 1
                if (self.source_idx >= self.bss) and (not len(self.queue)==0):
                    self.source_batch = self.sampler_get_kwargs()
                    self.bss = self.source_batch[0].shape[0]
                    self.source_idx = 0

        for k in list(batch_kwargs.keys()):
            batch_kwargs[k] = unet_kwarg_to_tensor(batch_kwargs[k],key=k,list_instead=True)
        
        batch_x = torch.stack(batch_x,dim=0)
        return batch_x, batch_kwargs, batch_info, batch_queue
    