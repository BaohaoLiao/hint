import subprocess
import os
import random
import string
from typing import Optional

#from fire import Fire
from pykrylov import Session, Task
import pykrylov
from pykrylov.util.consts import EXP_ID

from utils.pykrylov_ext import DeepspeedTask

os.environ["NO_PROXY"] = (
    "krylov,ams,ems,mms,localhost,127.0.0.1,.vip.ebay.com,.github.ebay.com,.tess.io,.corp.ebay.com,.ebayc3.com,.krylov-prod.svc"
)
PVC_NAME = "krylov-user-pvc-retina"
PVC_MOUNT_NAME = "retina-pvc"
PVC2_NAME = "krylov-user-pvc-retina-new"
PVC2_MOUNT_NAME = "retina-pvc2"
MASTER_PORT = 2020
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def train(
    model,
    script_path,
    output_dir,
    gpus_per_node,
    batch_size,
    per_device_batch_size,
):
    init_krylov_context()
    if model == "internvl":
        os.chdir(get_file_path("InternVL/internvl_chat"))
    elif model == "megatron":
        os.chdir(get_file_path("Megatron-LM"))
    elif model == "gpt":
        os.chdir(get_file_path("run"))

    output = subprocess.run(
        [
            "sh",
            get_file_path(script_path),
        ],
        env={
            **os.environ,
            "GPUS": str(gpus_per_node),
            "BATCH_SIZE": str(batch_size),
            "PER_DEVICE_BATCH_SIZE": str(per_device_batch_size),
            "OUTPUT_DIR": output_dir,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    )
    if output.returncode != 0:
        raise ValueError(f"Script exited with error {output.returncode}")
    subprocess.run(["chmod", "-R", "a+rw", output_dir])


def get_file_path(file_path: str) -> str:
    """Get the file path in the Krylov environment"""
    # if "KRYLOV_WF_HOME" in os.environ:
    #     task_folder = os.path.join(
    #         os.environ["KRYLOV_WF_HOME"], "src", os.environ["KRYLOV_WF_TASK_NAME"]
    #     )
    #     return os.path.join(task_folder, file_path)
    # else:
    #     return file_path
    return os.path.join(ROOT_DIR, file_path)
    

def init_krylov_context():
    init_krylov_common_context()
    if "KRYLOV_WS_NAME" not in os.environ:
        # Get params
        context = pykrylov.util.get_task_context()
        if "experiment_id" in context:
            experiment_id = context["experiment_id"]
            # These 2 lines make task logs viewable via experiment view on aihub
            pykrylov.util.set_global_context({EXP_ID: experiment_id})
            pykrylov.ems.experiment.update_experiment(
                experiment_id,
                runtime={"workflow": {"runId": os.environ["KRYLOV_WF_RUN_ID"]}},
            )


def init_krylov_common_context():
    krylov_data_dir = os.path.join(
        "/mnt", PVC_MOUNT_NAME, f"marmazur/cache{random.randint(1, 1000)}/"
    )
    os.environ["$KRYLOV_DATA_DIR"] = krylov_data_dir
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
    os.environ["PVC_MOUNT_PATH"] = os.path.join("/mnt", PVC_MOUNT_NAME)


def generate_token(user_name):
    session = Session(namespace="ebay")
    response = session.generate_service_account_token(service_account=user_name)
    print(response)


def submit_train(
    script_path: str = "ray/run.sh", #internvl_chat_gpt_oss/shell/internvl3_5_gpt_oss/internvl3_5_gpt_oss_20b_stage2_sft_mm.sh",
    output_dir: str = "./",  #/mnt/retina-pvc2/generalist/internvl35_gptoss_20B_test",
    docker_image: str = "hub.tess.io/image-understanding/lmm-training:latest-gpt",
    cpus_per_node: int = 2,
    gpus_per_node: int = 1,
    num_nodes: int = 2,
    gpu_model: str = "a100",
    memory: int = 16,
    batch_size: int = 16,
    per_device_batch_size: int = 1,
    labels: list = None,
    model: str = "gpt",
    experiment_name: str = "internvl_tuning",
    namespace: Optional[str] = "chatgpt",
):
    if batch_size % num_nodes != 0:
        raise ValueError(
            f"Batch size {batch_size} is not divisible by num_nodes {num_nodes}"
        )
    batch_size //= num_nodes
    print(f"Batch size: {batch_size} after dividing by num_nodes")

    assert model in ["internvl", "megatron", "gpt"]
    if labels is None:
        labels = []

    user_name = os.popen("whoami").read().rstrip()
    pykrylov.util.config.use_account(user_name, yubikey_required=True)

    master_name = model + "_" + "".join(random.choices(string.ascii_letters, k=8))
    master_service_name = master_name + "_svc"

    script_args = [
        model,
        script_path,
        output_dir,
        gpus_per_node,
        batch_size,
        per_device_batch_size,
    ]
    if num_nodes > 1:
        task = DeepspeedTask(
            train,
            args=script_args,
            name=master_name,
            main_service_port=MASTER_PORT,
            gpu_per_worker=gpus_per_node,
            num_workers=num_nodes,
        )
    else:
        task = Task(train, args=script_args)

    task.add_task_parameters(
        {
            "ems_project": "mnist-baliao",
            "experiment_name": model + "_training",
            "gpu_per_node": gpus_per_node,
            "num_nodes": num_nodes,
            "master_name": master_name,
            "master_service_name": master_service_name,
            "master_port": MASTER_PORT,
        }
    )

    task.set_image(docker_image)
    #task.mount_pvc(PVC_MOUNT_NAME, PVC_NAME, "tess38")
    #task.mount_pvc(PVC2_MOUNT_NAME, PVC2_NAME, "tess38")

    task.run_on_gpu(gpus_per_node, model=gpu_model)
    task.add_cpu(cpus_per_node)
    task.add_memory(memory)

    if model == "internvl" or model == "gpt":
        task.add_directory("run")
    elif model == "megatron":
        task.add_directory("Megatron-LM")
    else:
        raise (ValueError(f"Unrecognized model: {model}"))

    task.add_directory("utils")

    workflow = pykrylov.Flow(task)
    workflow.execution_parameters.add_execution_parameter("enableChooseCluster", "true")

    namespace = namespace or os.environ["KRYLOV_NAMESPACE"]
    session = Session(namespace=namespace, job_name="job-baliao-py-ba")

    project_name = "mnist-baliao"

    experiment_id = session.submit_experiment(
        workflow,
        project=project_name,
        experiment_name=experiment_name,
        labels=labels,
    )

    link1 = f"https://aip.vip.ebay.com/data/experiment-detail?projectName={project_name}&experimentId={experiment_id}"
    link2 = f"https://94.aihub.krylov.vip.ebay.com/projects/{project_name}/experiments/{experiment_id}"
    print(
        f"You can monitor progress and download result by visiting {link1} or {link2}"
    )


if __name__ == "__main__":
    submit_train()