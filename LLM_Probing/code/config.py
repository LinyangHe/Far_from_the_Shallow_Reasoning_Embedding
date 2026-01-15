import torch


class ExperimentConfig:
    """
    This class is used to store the configuration of an experiment.
    """
    def __init__(
        self,
        output_file_name=None,
        type_name=None,
        device=None,
        batch_size=32,
        sent_pooling='last',
        output_hidden_states=True,
        target_layer=None,
        pca_dim=-1,
        model_name=None,
        task=None,
        residual_mode=None,
        bow_mode=False,
        use_memmap=False,
        memmap_dir=None,
        filter_mode=False,
        blimp_bow_results_path="./results/blimp/blimp_base_bow_-1_bow.csv",
        residuals_blimp_prefix=None,
        residuals_comps_prefix=None,
        residuals_source_target_layers=((0, 6), (6, 20), (20, 30)),
        shuffle=True,
    ):
        self.output_file_name = output_file_name
        self.type_name = type_name
        self.device = device if device else torch.device("cpu")
        self.batch_size = batch_size
        self.sent_pooling = sent_pooling
        self.output_hidden_states = output_hidden_states
        self.target_layer = target_layer
        self.pca_dim = pca_dim
        self.model_name = model_name
        self.task = task
        self.residual_mode = residual_mode
        self.bow_mode = bow_mode
        self.use_memmap = use_memmap
        self.memmap_dir = memmap_dir
        self.filter_mode = filter_mode
        self.blimp_bow_results_path = blimp_bow_results_path
        if residuals_blimp_prefix is None and model_name:
            model_tag = model_name.split("/")[-1]
            residuals_blimp_prefix = f"/Data/BLiMP_{model_tag}/BLiMP_{model_tag}_-1"
        if residuals_comps_prefix is None and model_name:
            model_tag = model_name.split("/")[-1]
            residuals_comps_prefix = f"/Data/COMPS_{model_tag}/comps_base_{model_tag}_-1"
        self.residuals_blimp_prefix = residuals_blimp_prefix
        self.residuals_comps_prefix = residuals_comps_prefix
        self.residuals_source_target_layers = residuals_source_target_layers
        self.shuffle = shuffle

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
