import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

class ModelLoader:
    """
    load model and tokenizer and put them on the device
    """
    def __init__(self, model_name: str, cache_dir: str, hf_token: str, speech_mode: bool = True, asr_encoder: bool = True, baseline_mode: bool = True, use_gpu: bool = True, gpu_count: int = 1, dtype: torch.dtype = torch.bfloat16, untrained: bool = False):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.use_gpu = use_gpu
        self.gpu_count = gpu_count
        self.model = None
        self.tokenizer = None #for language model
        self.processor = None #for speech model
        self.speech_mode = speech_mode
        self.asr_encoder = asr_encoder 
        self.baseline_mode = baseline_mode
        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")
        self.dtype = dtype
        self.untrained = untrained

    def load(self):
        print(f"Loading model: {self.model_name} ..., device: {self.device}")
        if not self.speech_mode:
            if self.dtype == torch.qint8:
                self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name, 
                            cache_dir=self.cache_dir, 
                            token=self.hf_token,
                            trust_remote_code=True,
                            torch_dtype=torch.bfloat16,
                            device_map='auto'
                            )
                self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
            elif self.untrained:
                model_config = AutoConfig.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    token=self.hf_token,
                    trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_config(model_config)
                self.model = self.model.to(dtype=self.dtype)
                self.model = self.model.to(self.device)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                cache_dir=self.cache_dir, 
                token=self.hf_token,
                trust_remote_code=True,
                # torch_dtype=torch.bfloat16,
                torch_dtype=self.dtype,
                device_map='auto'
                )

            self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            cache_dir=self.cache_dir, 
            token=self.hf_token,
            trust_remote_code=True
            )
            # import pdb; pdb.set_trace()

        # pad token issue
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    print(f"pad token is set to eos token: {self.tokenizer.eos_token}")
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
                    print(f"pad token is set to <|endoftext|>")
        else: 
            if "qwen2-audio" in self.model_name.lower():
                from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
                if self.baseline_mode:
                    print("Baseline mode: no pretrained weights, just random initialization for Qwen2-Audio")
                    model_config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
                    self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
                    self.model = Qwen2AudioForConditionalGeneration.from_config(model_config)
                else:
                    self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
                    self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                        self.model_name, 
                        cache_dir=self.cache_dir,
                        trust_remote_code=True
                    )

            else:
                from transformers import AutoFeatureExtractor
                from transformers import AutoModel
                # self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
                # if 'wav2vec2' in self.model_name:
                #     from transformers import Wav2Vec2ForCTC
                #     speech_model = Wav2Vec2ForCTC
                    
                # elif 'hubert' in self.model_name:
                #     # from transformers import HubertForCTC
                #     from transformers import AutoModel
                #     speech_model = AutoModel
    
                # elif 'wavlm' in self.model_name:
                #     from transformers import WavLMModel
                #     speech_model = WavLMModel
                # else:
                #     raise ValueError(f"Model name {self.model_name} is not supported. Please check the model name.")

                if self.baseline_mode:
                    print("Baseline mode: no pretrained weights, just random initialization")
                    model_config = AutoConfig.from_pretrained(self.model_name)
                    self.processor = AutoFeatureExtractor.from_pretrained(self.model_name)
                    self.model = AutoModel.from_config(model_config).encoder if self.asr_encoder else AutoModel.from_config(model_config) # no pretrained weights, just random initialization              
                else:
                    self.processor = AutoFeatureExtractor.from_pretrained(self.model_name)
                    self.model = AutoModel.from_pretrained(
                        self.model_name, 
                        cache_dir=self.cache_dir,
                        trust_remote_code=True
                        # torch_dtype=torch.bfloat16
                        ).encoder if self.asr_encoder else AutoModel.from_pretrained(self.model_name, cache_dir=self.cache_dir, trust_remote_code=True)         

        # multi-gpu
        # if self.gpu_count > 1 and torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model)
        #     self.model = self.model.module
        
        # self.model.to(self.device, dtype=torch.bfloat16)
        print(f"Model loaded to device: {self.device} with dtype: {self.dtype}")
        
        if not self.speech_mode:
            return self.model, self.tokenizer
        else:
            return self.model, self.processor
