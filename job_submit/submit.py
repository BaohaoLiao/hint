import os
import random
import string
import argparse
import subprocess

import pykrylov
from job_submit.pykrylov_ext import DeepspeedTask


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_PORT = 2020


def parse_args():
    parser = argparse.ArgumentParser(description="CLI Configuration")
    parser.add_argument("script", type=str, required=True, help="Which script to run")
    parser.add_argument("--ems_project", type=str, default="mnist-baliao")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--cluster", type=str, default="tess137")
    parser.add_argument("--namespace", type=str, default="chatgpt-training-slc-a100")
    parser.add_argument("--image", type=str, default="hub.tess.io/baliao/rl:base")
    parser.add_argument("--cpu", type=int, default=16)
    parser.add_argument("--memory", type=int, default=64)
    parser.add_argument("--gpu_per_node", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--gpu_model", type=str, default="a100")
    parser.add_argument(
        "--rack_name", default=None, type=str, help="Specify which rack to use"
    )
    parser.add_argument("--job_name", default="job-baliao-py-ba", type=str)
    return parser.parse_args()


def init_krylov_common_context():
    krylov_data_dir = (
        f"/data/{os.environ['KRYLOV_NAMESPACE']}/data/{os.environ['KRYLOV_PRINCIPAL']}"
    )
    os.environ["KRYLOV_DATA_DIR"] = krylov_data_dir
    os.environ["HF_HOME"] = os.path.join(krylov_data_dir, ".lmm_cache/hf_cache")
    os.environ["HF_MODULES_CACHE"] = os.path.join(
        krylov_data_dir, ".lmm_cache/hf_module_cache"
    )
    os.environ["HF_DATASETS_CACHE"] = os.path.join(
        krylov_data_dir, ".lmm_cache/tmp/hf_data_cache"
    )
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(
        krylov_data_dir, ".lmm_cache/hf_cache"
    )
    os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(
        krylov_data_dir, ".lmm_cache/torch_cache"
    )
    # os.environ["TRITON_CACHE_DIR"] = os.path.join(krylov_data_dir, ".lmm_cache/.triton")
    # os.environ["PVC_MOUNT_PATH"] = os.path.join("/mnt", PVC_MOUNT_NAME)


def init_krylov_context():
    init_krylov_common_context()
    if "KRYLOV_WS_NAME" not in os.environ:
        # Get params
        context = pykrylov.util.get_task_context()
        if "experiment_id" in context:
            experiment_id = context["experiment_id"]
            # These 2 lines make task logs viewable via experiment view on aihub
            pykrylov.util.set_global_context(
                {pykrylov.util.consts.EXP_ID: experiment_id}
            )
            pykrylov.ems.experiment.update_experiment(
                experiment_id,
                runtime={"workflow": {"runId": os.environ["KRYLOV_WF_RUN_ID"]}},
            )
        return {
            "gpu_per_node": int(context["gpu_per_node"]),
            "num_nodes": int(context["num_nodes"]),
            "script": context["script"],
        }


def train():
    context = init_krylov_context()

    script_path = os.path.join(ROOT_DIR, context["script"])
    os.chmod(script_path, 755)

    output = subprocess.run(script_path, check=True)

    if output.returncode != 0:
        raise ValueError(f"Script exited with error {output.returncode}")


# subprocess.run(["chmod", "-R", "a+rw", output_dir])


def main(args):
    # Sanity check
    if args.rack_name is not None:
        assert args.rack_name in ["slc_slc03_01-0200_11_20", "slc_slc03_01-0200_12_20"]

    # Log in
    user_name = os.popen("whoami").read().rstrip()
    pykrylov.util.config.use_account(user_name, yubikey_required=True)

    master_name = "llm_" + "".join(random.choices(string.ascii_letters, k=8))
    master_service_name = master_name + "_svc"

    # Init pykrylov task
    if args.num_nodes > 1:
        task = DeepspeedTask(
            train,
            name=master_name,
            main_service_port=MASTER_PORT,
            gpu_per_worker=args.gpu_per_node,
            num_workers=args.num_nodes,
        )
    else:
        task = pykrylov.Task(train, args=[])

    # Task setting
    task.add_task_parameters(
        {
            "ems_project": args.ems_project,
            "experiment_name": args.exp_name,
            "gpu_per_node": args.gpu_per_node,
            "num_nodes": args.num_nodes,
            "master_name": master_name,
            "master_service_name": master_service_name,
            "master_port": MASTER_PORT,
        }
    )
    task.set_image(args.image)
    task.run_on_gpu(args.gpus_per_node, model=args.pu_model)
    task.add_cpu(args.cpu)
    task.add_memory(args.memory)
    task.add_file(args.script)
    task.add_execution_parameter("requireSameRack", "true")
    if args.rack_name is not None:
        task.add_execution_parameter(
            "nodeSelector", {"failure-domain.tess.io/rack": args.rack_name}
        )

    if args.cluster == "tess40":
        task.mount_nfs("mtrepo", "10.5.1.56", "/krylov_shared_volume/krylov_shared")
    if args.cluster == "tess137":
        task.mount_pvc("mtrepo", "nlp-ebert-01", args.cluster)
        task.mount_pvc("nushare2", "krylov-user-pvc-nlp-01", args.cluster)
    if args.cluster == "tess45":
        task.mount_pvc("nushare", "krylov-user-pvc-nlp-45", args.cluster)
        task.mount_pvc("mtrepo", "nlp-ebert-02", args.cluster)
        task.mount_pvc("nushare2", "krylov-user-pvc-nlp-01", args.cluster)
    if args.cluster == "tess38":
        task.mount_pvc("nushare2", "krylov-user-pvc-nlp-01", args.cluster)
        task.mount_pvc("nushare", "krylov-user-pvc-nlp-38", args.cluster)
        task.mount_pvc("mtrepo", "nlp-ebert-02", args.cluster)

    # Submit workflow
    workflow = pykrylov.Flow(task)
    workflow.execution_parameters.add_execution_parameter("enableChooseCluster", "true")

    session = pykrylov.Session(namespace=args.namespace, job_name=args.job_name)
    experiment_id = session.submit_experiment(
        workflow,
        project=args.ems_project,
        experiment_name=args.exp_name,
        labels=[],
    )

    link = f"https://aip.vip.ebay.com/data/experiment-detail?projectName={project_name}&experimentId={experiment_id}"
    print(f"You can monitor progress and download result by visiting {link}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
