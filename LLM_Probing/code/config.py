import torch

class ExperimentConfig:
    """
    This class is used to store the configuration of an experiment.
    """
    def __init__(
        self,
        output_file_name=None,
        file_id=None,
        field=None,
        ling_form=None,
        type_name=None,
        device=None,
        batch_size=32,
        sent_pooling='last',
        output_hidden_states=True,
        output_attentions=False,
        transfer_matrix=None,
        target_layer=None,
        speaker=None, # None for language model
        task_label_path=None,
        speech_data_path=None,
        speech_mode=False,
        pca_dim=-1,
        test_mode=False,
        model_name=None,
        task=None,
        gpu_classify=False,
        lang=None,
        embed_path=None, # save to .pkl file
        residual_mode=None,
        save_linear_weights=None,
        use_residual_cache=None,
        bow_mode=False,
        save_to_mat=False,
        filter_mode=False,
        mat_path=None, # save to .mat file
        thinking_mode_off=False, # for thinking mode, use the same config as test mode
        shuffle=True,
        blimp_residual_filter=False
    ):
        self.output_file_name = output_file_name
        self.file_id = file_id
        self.field = field
        self.ling_form = ling_form
        self.type_name = type_name
        self.device = device if device else torch.device("cpu")
        self.batch_size = batch_size
        self.sent_pooling = sent_pooling
        self.output_hidden_states = output_hidden_states
        self.transfer_matrix = transfer_matrix
        self.target_layer = target_layer
        self.output_attentions = output_attentions
        self.speaker = speaker
        self.task_label_path = task_label_path
        self.speech_data_path = speech_data_path
        self.speech_mode = speech_mode
        self.pca_dim = pca_dim
        self.test_mode = test_mode
        self.model_name = model_name
        self.task = task
        self.gpu_classify = gpu_classify
        self.lang = lang
        self.embed_path = embed_path
        self.residual_mode = residual_mode
        self.save_linear_weights = save_linear_weights
        self.use_residual_cache = use_residual_cache
        self.bow_mode = bow_mode
        self.save_to_mat = save_to_mat
        self.filter_mode = filter_mode
        self.mat_path = mat_path
        self.thinking_mode_off = thinking_mode_off
        self.shuffle = shuffle
        self.blimp_residual_filter = blimp_residual_filter

    def update(self, update_dict: dict):
        """
        Update the configuration with the given dictionary.
        """
        for k, v in update_dict.items():
            setattr(self, k, v)

    def __repr__(self):
        """
        Return a string representation of the configuration.
        """
        config_str = "ExperimentConfig:\n"
        for attr, value in self.__dict__.items():
            config_str += f"  {attr}: {value}\n"
        return config_str
