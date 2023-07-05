import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
from trident.utils.logging import get_logger

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


log = get_logger(__name__)


@hydra.main(version_base="1.2", config_path="configs/", config_name="config.yaml")
def main(cfg: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.train import train
    from src.utils.runner import extras, print_config

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    extras(cfg)
    # Init lightning datamodule

    # Pretty print config using Rich library
    if cfg.get("print_config"):
        print_config(cfg, resolve=True)

    # Train model
    return train(cfg)


if __name__ == "__main__":
    main()
