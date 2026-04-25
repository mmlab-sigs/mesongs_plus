import sys
import yaml
from simple_parsing import ArgumentParser
from splatwizard.config import PipelineParams
from splatwizard.model_zoo import CONFIG_CACHE
from splatwizard.modules.dataclass import TrainContext
from splatwizard.utils.misc import safe_state
from splatwizard.pipeline.rd_curve import run_rd_curve_pipeline
import pathlib
from loguru import logger

def main():
    parser = ArgumentParser(add_config_path_arg=True)
    parser.add_arguments(PipelineParams, dest="pipeline")
    parser.add_arguments(CONFIG_CACHE[0], dest="model_group")
    
    args = parser.parse_args(sys.argv[1:])
    
    mp = args.model_group.model
    pp = args.pipeline
    op = args.model_group.optim
    
    if pp.seed is not None:
        safe_state(pp.seed)
    
    if pp.seed is not None:
        safe_state(pp.seed)
    train_context = TrainContext()
    train_context.model = args.subgroups['model_group.model']

    if pp.output_dir is not None:
        train_context.base_output_dir = pathlib.Path(pp.output_dir)
        train_context.base_output_dir.mkdir(exist_ok=True, parents=True)
        train_context.output_dir = train_context.base_output_dir
        
    logger.add(train_context.output_dir / 'rd_curve.log')

    # Load rd_curve config from yaml if specified
    if hasattr(mp, 'yaml_path') and mp.yaml_path and mp.yaml_path != "":
        try:
            with open(mp.yaml_path, "r") as f:
                config_dict = yaml.safe_load(f)
            if "rd_curve_size_limits" in config_dict:
                mp.rd_curve_size_limits = config_dict["rd_curve_size_limits"]
                logger.info(f"Loaded rd_curve_size_limits from yaml: {mp.rd_curve_size_limits}")
            if "rd_curve_pruning_rates" in config_dict:
                mp.rd_curve_pruning_rates = config_dict["rd_curve_pruning_rates"]
                logger.info(f"Loaded rd_curve_pruning_rates from yaml: {mp.rd_curve_pruning_rates}")
            if "pruning_rates" in config_dict:
                mp.pruning_rates = config_dict["pruning_rates"]
                logger.info(f"Loaded pruning_rates from yaml: {mp.pruning_rates}")
            if "checkpoint_template" in config_dict:
                mp.checkpoint_template = config_dict["checkpoint_template"]
                logger.info(f"Loaded checkpoint_template from yaml: {mp.checkpoint_template}")
        except Exception as e:
            logger.warning(f"Failed to load rd_curve config from yaml {mp.yaml_path}: {e}")

    logger.info(f"{pp}")
    logger.info(f"{mp}")

    run_rd_curve_pipeline(pp, mp, op, train_context)

if __name__ == '__main__':
    sys.exit(main())

