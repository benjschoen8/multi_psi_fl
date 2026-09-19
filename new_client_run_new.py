"""
New-client experiment launcher (new_client_our_global_model.py / new_client_train_scratch.py untouched).

Why a wrapper: BaseNewClient copies only a few saved args, but the current data/client code also
reads args.noniid_partition and args.start_mapping_epoch -> AttributeError. This fills every missing
arg from the checkpoint (server_new.py saves the full args), then runs the original trainer class.

  python new_client_run_new.py --mode finetune --model_path=logs/<run>/GeFL_gan_pacfl_iid/server_checkpoints.pth \
      --device=cuda:0 --dataset_mode=single --new_client_epochs=30 --exp_timestamp=nc
  python new_client_run_new.py --mode scratch  ... (same args)

Output: <run>/new_client_<exp_timestamp>/<our_finetune_global|baseline>_<mode>_dataset/<Dataset>_newclient_history.csv
"""
import torch

from trainer.NewClient.BaseNewClient import get_base_parser

p = get_base_parser()
p.add_argument("--mode", choices=["finetune", "scratch"], default="finetune")
args = p.parse_args()

saved = torch.load(args.model_path, map_location="cpu", weights_only=False).get("args", {})
for k, v in saved.items():
    if not hasattr(args, k):
        setattr(args, k, v)
args.__dict__.setdefault("noniid_partition", "dirichlet")
args.__dict__.setdefault("start_mapping_epoch", 0)
if not saved.get("num_new_clients"):
    raise SystemExit("[new-client] checkpoint has num_new_clients=0: rerun FL with --num_new_clients=1 (or more)")

if args.mode == "finetune":
    from new_client_our_global_model import FinetuneGlobalModelTrainer as T
    T(mode_name="our_finetune_global", args=args).run()
else:
    from new_client_train_scratch import ScratchTrainer as T
    T(mode_name="baseline", args=args).run()
