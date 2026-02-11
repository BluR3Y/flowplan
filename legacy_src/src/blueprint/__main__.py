import argparse
import logging
from blueprint.config.loader import load_config
from blueprint.core.models import Config
from blueprint.core.pipeline import Pipeline
from blueprint.tools.profiler import profile_file
import json

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    ap = argparse.ArgumentParser("flowplan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    
    run_p = sub.add_parser("run")
    run_p.add_argument("--config", required=True)
    run_p.add_argument("--profile")
    
    profile_p = sub.add_parser("profile", help="Generate schema from a source file")
    profile_p.add_argument("--file", required=True, help="Path to Excel or CSV file")
    profile_p.add_argument("--sheet", default=0, help="Sheet name or index (default 0)")
    
    args = ap.parse_args()
    
    if args.cmd == "run":
        raw_cfg = load_config(args.config, args.profile)
        cfg = Config(raw_cfg)
        
        pipe = Pipeline(cfg)
        pipe.run()
    elif args.cmd == "profile":
        try:
            result = profile_file(args.file, args.sheet)
            print(json.dumps(result, indent=2))
            return 0
        except Exception as e:
            logging.error(f"Profiling failed: {e}")
            return 1

if __name__ == "__main__":
    main()