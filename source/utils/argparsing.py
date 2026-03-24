
import argparse
import json
import sys
from pathlib import Path
from functools import partial
from collections import OrderedDict
from source.utils.mixed import load_json_to_dict_list, save_dict_list_to_json, longest_common_substring, bracket_glob_fix
import copy


to_list_if_str = lambda x: [x] if isinstance(x,str) else x

def get_ckpt_name(s,saves_folder="./saves/",return_multiple_matches=False):
    if ";" in s:
        return sum([to_list_if_str(get_ckpt_name(x,saves_folder,return_multiple_matches)) for x in s.split(";")],[])
    s_orig = copy.copy(s)
    if len(s)==0:
        return s
    assert not s.find("./")>=0, "name_match_str is already relative to saves_folder. Do not use ./ in name_match_str."
    # converts to format: */*.pt
    print("Trying to find ckpt name from s=",s)
    num_sep = s.count("/")
    if num_sep==0:
        s = s+"/ckpt_*.pt"
    elif num_sep>=1:
        if not s.endswith(".pt"):
            s = s+"/ckpt_*.pt"
    if s.find("*") >= 0:
        matching_paths = list(Path(saves_folder).glob(bracket_glob_fix(s)))
        print(f"Found {len(matching_paths)} matches for s={s}.")
        if len(matching_paths) == 0:
            print("Raising ValueError.")
            raise ValueError(f"No models match the expression: {s_orig}, consider using a starred expression. The string was modified to: {s}.")
        elif len(matching_paths)==1:
            s = str(matching_paths[0])
        else:
            if return_multiple_matches:
                s = sorted([str(x) for x in matching_paths])
            else:
                raise ValueError("Multiple models match the expression. Be more specific than name_match_str="+s+"\n matches listed below: \n"+str([str(x) for x in matching_paths]))
    else:
        s = str(s)
        assert Path(s).exists(), "model path does not exist. Use starred expressions to search s="+s
    return s

def list_wrap_type(t):
    def list_wrap(x):
        if isinstance(x,str):
            if x.find(";")>=0:
                return [t(y) for y in x.split(";") if len(y)>0]
            else:
                return t(x)
        else:
            return t(x)
    return list_wrap

def load_defaults(idx=0,
                  ordered_dict=False,
                  filename="configs/args_default.json"):
    default_path = Path(__file__).parent.parent.parent/filename
    if ordered_dict:
        args_dicts = json.loads(default_path.read_text(), object_pairs_hook=OrderedDict)    
    else:
        args_dicts = json.loads(default_path.read_text())
    args_dict = {}
    for k,v in args_dicts.items():
        if isinstance(v,dict):
            for k2,v2 in v.items():
                args_dict[k2] = v2[idx]
        else:
            args_dict[k] = v[idx]
    return args_dict

def compare_strs(str1,str2,operator):
    if operator=="==":
        out = str1==str2
    elif operator==">=":
        out = str1>=str2
    elif operator=="<=":
        out = str1<=str2
    elif operator=="<" or operator=="<<":
        out = str1<str2
    elif operator==">" or operator==">>":
        out = str1>str2
    else:
        raise ValueError(f"operator={operator} not supported.")
    return out

def str2bool(v):
    if isinstance(v, bool):
        return v
    elif isinstance(v, (int,float)):
        return bool(v)
    elif isinstance(v, str):
        if v.lower() in ["yes", "true", "t", "y", "1"]:
            return True
        elif v.lower() in ["no", "false", "f", "n", "0"]:
            return False
        else:
            raise argparse.ArgumentTypeError("Cannot convert string: {} to bool".format(v))
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
    
def str_with_semicolon_version(v):
    """
    Converts strings of the type 
    from       |   to
    -----------------------
    'm[v1;v2]'         -> 'm[v1];m[v2]'
    'm1[v1];m2[v1;v2]' -> 'm1[v1];m2[v1];m2[v2]'
    'm[v1;v2]+p1'      -> 'm[v1]+p1;m[v2]+p1'
    """
    assert v.count("[")==v.count("]"), f"v={v} has mismatched brackets."
    if v.find(";")<0:
        return v
    #replace all ";" inside brackets with ":"
    v2 = ""
    open_brackets = 0
    for s_idx in range(len(v)):
        if v[s_idx]=="[":
            open_brackets += 1
        elif v[s_idx]=="]":
            open_brackets -= 1
        if v[s_idx]==";":
            if open_brackets>0:
                v2 += ":"
            else:
                v2 += ";"
        else:
            v2 += v[s_idx]
    #split by ";"
    v_out = ""
    for s in v2.split(";"):
        s1 = s[:s.find("[")]
        bracket_string = s[s.find("["):s.find("]")+1]
        s3 = s[s.find("]")+1:]
        for s2 in bracket_string[1:-1].split(":"):
            v_out += s1+"["+s2+"]"+s3+";"
    v_out = v_out[:-1]
    return v_out.split(";")

class TieredParser():
    def __init__(self,name="args",
                 tiers_dict={"modified_args": 0,
                                  "commandline": 1,
                                  "name_plus": 2,
                                  "name_versions": 3, 
                                  "name_root": 4, 
                                  "defaults_version": 5, 
                                  "defaults_current": 6},
                    key_to_type={"origin": dict,
                                 "name_match_str": lambda x: get_ckpt_name(x,return_multiple_matches=True),
                                 "model_name": str_with_semicolon_version,
                                 "gen_setup": str_with_semicolon_version}):
        self.tiers_dict = tiers_dict
        assert name in ["args","sample_opts"], f"name={name} not supported."
        self.name_key = {"args": "model_name","sample_opts": "gen_setup"}[name]
        self.id_key = {"args": "model_id","sample_opts": "gen_id"}[name]
        self.filename_def   = "configs/"+name+"_default.json"
        self.filename_model = "configs/"+name+"_configs.json"
        self.filename_ids   = "configs/"+name+"_ids.json"
        self.repo_root = Path(__file__).parent.parent.parent

        self.defaults_func = partial(load_defaults,filename=self.filename_def)
        self.descriptions = self.defaults_func(idx=1)

        self.parser = argparse.ArgumentParser()
        self.type_dict = {}
        for k, v in self.defaults_func().items():
            v_hat = v
            if k in key_to_type.keys():
                t = key_to_type[k]
            else:
                t = get_type_from_default(v)
            if isinstance(v, str):
                if v.endswith(","):
                    v_hat = v[:-1]
            self.parser.add_argument(f"--{k}", 
                                     default=v_hat, 
                                     type=t, 
                                     help=self.get_description_from_key(k))
            self.type_dict[k] = t

    def construct_args(self,tiers,tiers_dict=None,tiers_for_origin=list(range(4))):
        if tiers_dict is None:
            tiers_dict = self.tiers_dict
        tier_numbers = sorted(list(tiers_dict.values()),reverse=True)
        tiers_dict_inv = {v: k for k,v in tiers_dict.items()}
        origin = {}
        args = {}
        for tier_num in tier_numbers:
            tier_name = tiers_dict_inv[tier_num]
            for k,v in tiers[tier_name].items():
                if k not in self.type_dict.keys():
                    raise ValueError(f"Recieved unrecognized argument k={k} from source: {tier_name}. Closest known matches: {get_closest_matches(k,self.type_dict.keys(),n=3)}")
            args.update(tiers[tier_name])
            origin.update({k: tier_name for k in tiers[tier_name].keys()})
        tfo = [tiers_dict_inv[k] for k in tiers_for_origin]
        args["origin"] = {k: v for k,v in origin.items() if v in tfo}
        return args

    def get_command_line_args(self,alt_parse_args=None,use_parser=False):
        
        if alt_parse_args is None:
            commandline_list = sys.argv[1:]
        else:
            commandline_list = alt_parse_args
        if len(commandline_list)==1:
            if commandline_list[0]=="--help":
                self.parser.print_help()
                sys.exit()
        assert len(commandline_list)%2==0, f"commandline_list={commandline_list} must have an even number of elements."
        assert all([x.startswith("--") for x in commandline_list[::2]]), f"All even elements of commandline_list={commandline_list} must start with --."
        assert all([not x.startswith("--") for x in commandline_list[1::2]]), f"All odd elements of commandline_list={commandline_list} must not start with --."
        if use_parser:
            args = self.parser.parse_args(commandline_list).__dict__
        else:
            args = {k[2:]: v for (k,v) in zip(commandline_list[::2],commandline_list[1::2])}
        return args

    def get_name_based_args(self,name):
        assert isinstance(name,str), f"name={name} not a valid type."
        if name.find(";")>=0:
            return {}, {}, {}
        name_based_args = load_json_to_dict_list(str(Path(__file__).parent.parent.parent/self.filename_model))
        if "+" in name:
            plus_names = name.split("+")[1:]
            root_name = name.split("+")[0]
        else:
            plus_names = []
            root_name = name
        ver_names = []
        if ("[" in name) and ("]" in name):
            for _ in range(name.count("[")):
                idx0 = root_name.find("[")
                idx1 = root_name.find("]")
                if idx0<0 or idx1<0:
                    raise ValueError(f"name={name} has mismatched brackets.")
                ver_names.append(root_name[idx0+1:idx1])
                root_name = root_name[:idx0] + root_name[idx1+1:]
        #check that we are not using illegal version names (keys)
        all_keys = self.defaults_func().keys()
        for ver_name in ver_names:
            if ver_name in all_keys:
                raise ValueError(f"ver_name={ver_name} is not a valid version name because it is already a key for args.")
        if root_name in name_based_args.keys():
            root_name_args = name_based_args[root_name]
        else:
            raise ValueError(f"name={root_name} not found in name_based_args")
        ver_name_args = {}
        if len(ver_names)>0:
            assert "versions" in name_based_args[root_name].keys(), f"name={root_name} does not have versions."
            assert all([k in name_based_args[root_name]["versions"].keys() for k in ver_names]), f"ver_names={ver_names} not found in name_based_args[root_name]['versions'].keys()={name_based_args[root_name]['versions'].keys()}"
            for k,v in name_based_args[root_name]["versions"].items():
                if k in ver_names:
                    ver_name_args.update(v)
        plus_name_args = {}
        for pn in plus_names:
            assert "+"+pn in name_based_args.keys(), f"plus_name={pn} not found in name_based_args."
            plus_name_args.update(name_based_args["+"+pn])
        if "versions" in root_name_args.keys():
            del root_name_args["versions"]
        return root_name_args, plus_name_args, ver_name_args

    def get_args(self,alt_parse_args=None,modified_args={}):
        tiers = {k: {} for k in self.tiers_dict.keys()}

        tiers["modified_args"] = modified_args # top default priority
        tiers["commandline"] = self.get_command_line_args(alt_parse_args)
        tiers["defaults_current"] = self.defaults_func()
        name = self.construct_args(tiers)[self.name_key]
        root_name_args, plus_name_args, ver_name_args = self.get_name_based_args(name=name)
        tiers["name_plus"] = plus_name_args
        tiers["name_versions"] = ver_name_args
        tiers["name_root"] = root_name_args
        tiers["defaults_version"] = self.defaults_func()
        #find version only after all other args are set
        args = self.construct_args(tiers)
        #map to the correct types
        args = self.parse_types(args)
        if any([isinstance(v,list) for (k,v) in args.__dict__.items()]):
            modified_args_list = []
            num_modified_args = 1
            for k,v in args.__dict__.items():
                if isinstance(v,list):
                    #if len(v)>1:
                    num_modified_args *= len(v)
                    if num_modified_args>100:
                        raise ValueError(f"Too many modified args. num_modified_args={num_modified_args}")
                    if len(modified_args_list)==0:
                        modified_args_list.extend([{k: v2} for v2 in v])
                    else:
                        modified_args_list = [{**d, k: v2} for d in modified_args_list for v2 in v]
            if len(modified_args_list)>0:
                return modified_args_list
        setattr(args,self.id_key,self.get_unique_id(args))
        return args
    
    def parse_types(self, args):
        #args_dict = {k: v if isinstance(v,list) else self.type_dict[k](v) for k,v in args.items()}
        args_dict = {}
        for k,v in args.items():
            try:
                if isinstance(v,list):
                    args_dict[k] = [self.type_dict[k](v2) for v2 in v]
                else:
                    args_dict[k] = self.type_dict[k](v)
            except:
                print(f"Error parsing key={k} with value={v} and type={self.type_dict[k]}.")
                raise
        args = argparse.Namespace(**args_dict)
        return args
    
    def get_description_from_key(self, k):
        if k in self.descriptions.keys():
            return self.descriptions[k]
        else:
            return ""

    def _local_id_file_paths(self):
        saves_root = self.repo_root / "saves"
        if not saves_root.exists():
            return []
        if self.id_key=="model_id":
            pattern = "**/args.json"
        else:
            pattern = "**/sample_opts.json"
        return sorted(saves_root.glob(pattern))

    def load_and_format_id_dict(self,return_type="dict",include_legacy_global=True):
        id_list = []
        for file_path in self._local_id_file_paths():
            loaded = load_json_to_dict_list(str(file_path))
            if isinstance(loaded,list):
                id_list.extend([item for item in loaded if isinstance(item,dict) and self.id_key in item])
            elif isinstance(loaded,dict) and self.id_key in loaded:
                id_list.append(loaded)

        legacy_path = self.repo_root / self.filename_ids
        if include_legacy_global and legacy_path.exists():
            legacy_loaded = load_json_to_dict_list(str(legacy_path))
            if isinstance(legacy_loaded,list):
                id_list.extend([item for item in legacy_loaded if isinstance(item,dict) and self.id_key in item])
            elif isinstance(legacy_loaded,dict) and self.id_key in legacy_loaded:
                id_list.append(legacy_loaded)

        if return_type=="list":
            return id_list
        elif return_type=="dict":
            id_dict = {}
        elif return_type=="ordereddict":
            id_dict = OrderedDict()
        else:
            raise ValueError(f"return_type={return_type} not supported. must be 'list', 'dict', or 'ordereddict'.")
        for item in id_list:
            id_of_item = item[self.id_key]
            id_dict[id_of_item] = item
        return id_dict
    
    def is_unique_id(self, id):
        id_dict = self.load_and_format_id_dict()
        return id not in id_dict.keys()

    def get_unique_id(self, args):
        id_dict = self.load_and_format_id_dict(include_legacy_global=False)
        id = args.__dict__[self.id_key]
        for k,v in args.__dict__.items():
            id = id.replace(f"[{k}]",str(v))
        if id.find("*")>=0:
            if len(id_dict)==0:
                id = id.replace("*","0")
            else:
                for i in range(len(id_dict)):
                    if id.replace("*",str(i)) not in id_dict.keys():
                        id = id.replace("*",str(i))
                        break
        assert id not in id_dict.keys(), f"id={id} already exists in id_dict.keys(). use a starred expression to get a unique id."
        return id
    
def get_type_from_default(default_v):
    assert isinstance(default_v,(float,int,str,bool)), f"default_v={default_v} is not a valid type."
    if isinstance(default_v, str):
        assert default_v.find(";")<0, f"semicolon not supported in default arguments"
        def t2(x):
            assert isinstance(x,str), f"x={x} is not a valid type."
            if x.endswith(","):
                return str(x[:-1])
            else:
                return str(x)
    else:
        t2 = type(default_v)
    t = list_wrap_type(str2bool if isinstance(default_v, bool) else t2)
    return t

def get_closest_matches(k, list_of_things, n=3):
    """finds the n closest matched between a specified string, k, and the 
    keys of the type_dict. Closeness is measured by intersection 
    (len(intersection) where intersection is the longest common substring) 
    over union (len(k1) + len(k2) - len(intersection))."""
    iou_per_key = {}
    for k2 in list_of_things:
        intersection = longest_common_substring(k,k2)
        iou_per_key[k2] = len(intersection)/(len(k)+len(k2)-len(intersection))
    return [a[0] for a in sorted(iou_per_key.items(), key=lambda x: x[1], reverse=True)[:n]]

def load_existing_args(path_or_id,
                  name_key="args"):
    tp = TieredParser(name_key)
    if str(path_or_id).endswith(".json"):
        args_loaded = load_json_to_dict_list(str(Path(path_or_id)))
        if isinstance(args_loaded,list):
            assert len(args_loaded)==1, f"Expected len(args_loaded)==1, but len(args_loaded)={len(args_loaded)} for path_or_id={path_or_id}."
            args_loaded = args_loaded[0]
    else:
        id_dict = tp.load_and_format_id_dict()
        assert path_or_id in id_dict.keys(), f"path_or_id={path_or_id} not found in id_dict.keys(). Closest matches: {get_closest_matches(path_or_id,id_dict.keys(),n=3)}"
        args_loaded = id_dict[path_or_id]
    return argparse.Namespace(**args_loaded)

def overwrite_existing_args(args,delete_instead_of_overwrite=False):
    local_path, _ = save_args(args,dry=True)
    if hasattr(args,"model_name"):
        tp = TieredParser("args")
    elif hasattr(args,"gen_setup"):
        tp = TieredParser("sample_opts")
    else:
        raise ValueError(f"Expected args to contain either model_name or gen_setup.")
    dict_list = load_json_to_dict_list(local_path)
    if isinstance(dict_list,dict):
        dict_list = [dict_list]
    if not isinstance(dict_list,list):
        dict_list = []
    found = False
    for i in range(len(dict_list)):
        if isinstance(dict_list[i],dict) and dict_list[i].get(tp.id_key)==args.__dict__[tp.id_key]:
            if delete_instead_of_overwrite:
                del dict_list[i]
            else:
                dict_list[i] = args.__dict__
            found = True
            break
    if (not found) and (not delete_instead_of_overwrite):
        dict_list.append(args.__dict__)
    save_dict_list_to_json(dict_list,local_path,append=False)

def save_args(args, local_path=None, global_path=None, dry=False, do_nothing=False, append_local=True):
    if do_nothing:
        return local_path, global_path
    if hasattr(args,"model_name"):
        if local_path is None:
            local_path = args.save_path+"/args.json"
        else:
            assert local_path.endswith(".json"), f"local_path={local_path} must end with .json"
    elif hasattr(args,"gen_setup"):
        if local_path is None:
            local_path = str(Path(args.default_save_folder)/"sample_opts.json")
        else:
            assert local_path.endswith(".json"), f"local_path={local_path} must end with .json"
    else:
        raise ValueError(f"Expected args to contain either model_name or gen_setup.")
    
    if global_path is not None:
        assert global_path.endswith(".json"), f"global_path={global_path} must end with .json"
    assert Path(local_path).parent.exists(), f"Path(local_path).parent={Path(local_path).parent} does not exist."
    if not dry:
        save_dict_list_to_json([args.__dict__],str(local_path),append=append_local)
    return local_path, global_path
