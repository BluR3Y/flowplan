import argparse
import logging
from flowplan.config.loader import load_config
from flowplan.core.models import Config
from flowplan.core.pipeline import Pipeline

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    ap = argparse.ArgumentParser("flowplan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    
    runp = sub.add_parser("run")
    runp.add_argument("--config", required=True)
    runp.add_argument("--profile")
    
    args = ap.parse_args()
    
    if args.cmd == "run":
        raw_cfg = load_config(args.config, args.profile)
        cfg = Config(raw_cfg)
        
        pipe = Pipeline(cfg)
        pipe.run()

if __name__ == "__main__":
    main()