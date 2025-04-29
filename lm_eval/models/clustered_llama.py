from lm_eval.models.huggingface import HFLM
from lm_eval.models.llama2_a100 import LlamaForCausalLM as LlamaForCausalLMClustered  # your custom model
from lm_eval.api.registry import register_model
import torch
import os
from lm_eval.models.llama2_a100 import LlamaAttention as LlamaClusteredAttention
from transformers import AutoTokenizer
"""
def load_checkpoint_lm_eval(folder_name, epoch=None, file_name=None):
    Device-agnostic checkpoint loading for lm-eval compatibility.
    Returns the state dict without forcing it to any specific device.
    
    current = os.getcwd()
    try:
        if epoch is not None:
            checkpoint_path = os.path.join(current, 'checkpoint', folder_name, str(epoch))
        else:
            checkpoint_path = os.path.join(current, 'checkpoint', folder_name)
        
        if file_name is None:
            file_name = 'ckpt.t7'
            
        full_path = os.path.join(checkpoint_path, file_name)
        # Load to CPU first - let lm-eval handle device mapping
        state = torch.load(full_path, map_location='cpu')
        return state
    finally:
        os.chdir(current)  # Ensure we always restore the working directory
"""

def load_checkpoint_lm_eval(folder_name, device):
    """
    Device-agnostic checkpoint loading for lm-eval compatibility.
    Returns the state dict without forcing it to any specific device.
    """
    file_name = 'ckpt.t7'
    full_path = os.path.join(folder_name, file_name)
    # Load to CPU first - let lm-eval handle device mapping
    state = torch.load(full_path, map_location=device)
    return state


@register_model("clustered-llama")
class CustomLlamaEval(HFLM):
    def __init__(
        self,
        pretrained='checkpoint',  # default checkpoint path
        cluster_dir='cluster',    # default cluster directory
        dist_type='inner_product',
        **kwargs
    ):
        self.cluster_dir = cluster_dir
        self.dist_type = dist_type
        print("Initializing clustered llama")
        pretrained_model = LlamaForCausalLMClustered.from_pretrained(pretrained).half().to('cuda:0')
        tokenizer = AutoTokenizer.from_pretrained("checkpoint")
        #pretrained_model.set_cluster_info(centroid, self.dist_type)
        super().__init__(pretrained=pretrained_model, tokenizer=tokenizer, **kwargs)
        #super().__init__(pretrained=pretrained, **kwargs)
        self.set_cluster_model()

    def set_cluster_model(self):
        # Load base model first
        #self.model = LlamaForCausalLMClustered.from_pretrained(
        #    self.pretrained,
        #    torch_dtype=torch.float16,
        #    device_map="auto"
        #)
        print("Setting Cluster for Model from cluster directory %s" % self.cluster_dir)
        
        # Load cluster info
        state = load_checkpoint_lm_eval(self.cluster_dir, device=self.device)
        centroid = state['centroid']
        self.model.set_cluster_info(centroid, self.dist_type)

        return self.model
    

