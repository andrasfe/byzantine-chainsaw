"""Main entry point for the quantum Byzantine detection system."""

import argparse
import logging
import sys
from pathlib import Path

from config.experiment_config import ExperimentConfig
from experiments.experiment_runner import ExperimentRunner

def setup_logging(log_level: str, output_dir: str):
    """Setup logging configuration"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(output_dir) / 'experiment.log')
        ]
    )

def main():
    parser = argparse.ArgumentParser(
        description="Quantum-Enhanced Byzantine Detection in Federated Learning"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running experiment"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = ExperimentConfig.from_yaml(args.config)
        
        # Override log level if specified
        if args.log_level:
            config.log_level = args.log_level
        
        # Setup logging
        setup_logging(config.log_level, config.output_dir)
        logger = logging.getLogger(__name__)
        
        if args.dry_run:
            logger.info("Configuration validated successfully")
            logger.info(f"Experiment: {config.experiment_name}")
            logger.info(f"Dataset: {config.dataset}")
            logger.info(f"Federated Learning: {config.federated}")
            logger.info(f"Quantum Config: {config.quantum}")
            logger.info(f"Detection Config: {config.detection}")
            logger.info(f"Attack Config: {config.attack}")
            return
        
        # Run experiment
        logger.info(f"Starting experiment: {config.experiment_name}")
        runner = ExperimentRunner(config)
        runner.run()
        logger.info("Experiment completed successfully")
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Experiment failed: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()