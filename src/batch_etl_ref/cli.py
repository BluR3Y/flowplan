import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger("batch_etl")

from .config import load_config
from .sources.registry import get_adapter
from .engine.compile import Compiler

def main(argv=None):
    ap = argparse.ArgumentParser("batch-etl")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run_cmd = sub.add_parser("run")
    run_cmd.add_argument("--config", required=True)
    run_cmd.add_argument("--profile")
    
    args = ap.parse_args(argv)

    if args.cmd == "run":
        try:
            # 1. Load Config
            cfg = load_config(args.config, args.profile)

            # 2. Load Sources
            frames = {}
            for src in cfg.sources:
                adapter = get_adapter(src, cfg.schema_aliases)
                frames.update(adapter.load_tables())

            # 3. Compile
            compiler = Compiler(frames, cfg.schema_aliases)
            compiled = {}
            for tgt in cfg.compile_targets:
                compiled[tgt["name"]] = compiler.compile_target(tgt)
            
            # 4. Export
            
        except Exception as e:
            log.exception("Pipeline failed")
            sys.exit(1)

if __name__ == "__main__":
    main()