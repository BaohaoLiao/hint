import os

from pykrylov.contrib.constants.contents import (
    BASIC_VARS_FORMAT,
    IP_VAR_FORMAT,
    MASTER_IP_DEF,
    MASTER_LOCAL_DEF,
    MASTER_VAR_FORMAT,
    SVC_IP_DEF,
)
from pykrylov.contrib.task_helpers.content_generator import ContentGenerator
from pykrylov.contrib.tasks.parallel_gpu_task import ParallelGpuTask

HOME_TMP = """
export HOME="/tmp"
"""


def get_port_env_var_name(s_service_name, s_task_name):
    """
    Get the Krylov environment variable name of PORT list.
    :param s_service_name: sanitized service name.
    :param s_task_name: sanitized task name.
    :return: Krylov environment variable name of PORT list.
    """
    return "SVC_" + s_task_name + "_" + s_service_name + "_PORT_LIST"


def get_deepspeed_vars(gpus_per_worker, s_task_name, s_service_name):
    ip_env_var_name = ContentGenerator.get_ip_env_var_name(s_service_name, s_task_name)
    port_env_var_name = get_port_env_var_name(s_service_name, s_task_name)
    content = f"""
    IFS=',' read -r -a ip_array <<< "${ip_env_var_name}"
    IFS=',' read -r -a port_array <<< "${port_env_var_name}"

    export DEEPSPEED_NUM_NODES="${{#ip_array[@]}}"
    export DEEPSPEED_NUM_GPUS="{gpus_per_worker}"
    export DEEPSPEED_HOSTFILE="/tmp/hostfile"
    export DEEPSPEED_MASTER_ADDR="${{ip_array[0]}}"
    export DEEPSPEED_MASTER_PORT="${{port_array[0]}}"
    export DEEPSPEED_NODE_RANK="${{task_index}}"
    """
    return content


def get_deepspeed_hostfile_script_content(gpus_per_worker, s_task_name, s_service_name):
    content = (
        get_deepspeed_vars(gpus_per_worker, s_task_name, s_service_name)
        + f"""
    for ip in "${{ip_array[@]}}"; do
        echo "$ip slots={gpus_per_worker}" >> "$DEEPSPEED_HOSTFILE"
    done

    echodate "Deepspeed hostfile created at $DEEPSPEED_HOSTFILE with content:"
    echodate "$(cat $DEEPSPEED_HOSTFILE)"
    """
    )
    return content


class DeepspeedTask(ParallelGpuTask):
    def __init__(
        self,
        obj,
        args=None,
        docker_image=None,
        gpu_model=None,
        gpu_per_worker=None,
        num_workers=None,
        name=None,
        main_service_port=2020,
        extra_service_ports=None,
        extra_opts=None,
        optional_py_file=None,
        optional_py_file_args=None,
        extra_cmd_before_start=None,
        **kwargs,
    ):
        """
        DeepspeedTask constructor.
        :param obj: A python function or the entry point python file.
        :param args: Arguments to the task object function or the python file;
            list, str or dict, e.g, could be [2,3] if your function takes two int arguments, or '--a=2 --b=3'
            if your python script takes command line arguments, or {'a': 2, 'b': 'sgd'} but your function needs to
            use json.loads to parse the arguments.
        :param docker_image: URL of the docker image to be used during execution.
        :param gpu_model: GPU model, currently supports p100, v100, v100g1, v100g2, v100g3.
        :param gpu_per_worker: number of GPUs per worker.
        :param num_workers: number of parallel workers, int, e.g. 3. This refers to nodes/machines/boxes, NOT cards.
        :param name: Name of the task. If none provided, obj.__name__ or the python file name will be set as task name.
        :param main_service_port: port used for tcp. By default: 2020.
        :param extra_service_ports: list of extra ports if needed. For example, port 22 for ssh.
        :param extra_opts: extra opts for the framework.
        :param optional_py_file: Optional other python file which needs to run separately before mpirun,
                                 such as tensorpack data serving process.
        :param optional_py_file_args: Arguments for the other python file which needs to run separately.
        :param extra_cmd_before_start: A command string which needs to run before the optional py file and mpirun.
        """
        super(DeepspeedTask, self).__init__(
            obj,
            args,
            docker_image,
            gpu_model,
            gpu_per_worker,
            num_workers,
            name,
            main_service_port,
            extra_service_ports,
            extra_opts,
            optional_py_file=optional_py_file,
            optional_py_file_args=optional_py_file_args,
            extra_cmd_before_start=extra_cmd_before_start,
            **kwargs,
        )
        if optional_py_file:
            self.add_file(optional_py_file)

    def prepare_prestart(self, obj, user_args, *args, **kwargs):
        """
        Prepare the prestart shell script for the DeepspeedTask.
        The script creates deepspeed hostfile.

        :param obj: user passed python function or python file.
        :param user_args: arguments to the python function or python file.
        :param args: other args for possible future child class extendability.
        :param kwargs: other kwargs for possible future child class extendability.
        :return:
        """
        prestart_script = ContentGenerator.write_content_to_file(
            self.object_manager.task_file,
            ".prestart.sh",
            ContentGenerator.get_bash_header()
            + ContentGenerator.get_echo_start()
            # due to read only root filesystem in tess 137
            + HOME_TMP
            + get_deepspeed_hostfile_script_content(
                kwargs.get("gpu_per_worker"), self.s_task_name, self.s_service_name
            )
            + ContentGenerator.get_echo_end(),
        )
        self.add_prestart(prestart_script)

    def prepare_start(self, obj, user_args, *args, **kwargs):
        """
        Prepare the wrapper start shell script for the DeepspeedTask,

        :param obj: user passed python function or python file.
        :param user_args: arguments to the python function or python file.
        :param args: other args for possible future child class extendability.
        :param kwargs: other kwargs for possible future child class extendability.
        :return:
        """
        python_file = os.path.basename(self.object_manager.task_file)
        num_workers = kwargs.get("num_workers")
        basic_variables_def = BASIC_VARS_FORMAT.format(
            python_file, num_workers, kwargs.get("gpu_per_worker")
        )
        tcp_port = kwargs.get("main_service_port")
        master_var_def = MASTER_VAR_FORMAT.format(tcp_port)

        ip_def = ""
        if tcp_port is not None:
            env_var_name = ContentGenerator.get_ip_env_var_name(
                self.s_service_name, self.s_task_name
            )
            ip_var_def = IP_VAR_FORMAT.format(env_var_name)
            ip_def = ip_var_def + SVC_IP_DEF + MASTER_IP_DEF
        else:
            ip_def = MASTER_LOCAL_DEF
        if num_workers < 1:
            raise ValueError("num_workers %s not supported" % num_workers)

        py_cmd = ContentGenerator.get_simple_py_cmd(
            python_file, self.object_manager.user_args
        )

        ContentGenerator.write_content_to_file(
            self.object_manager.task_file,
            ".start.sh",
            ContentGenerator.get_bash_header()
            + ContentGenerator.get_echo_start()
            + basic_variables_def
            + master_var_def
            + ip_def
            + get_deepspeed_vars(
                kwargs.get("gpu_per_worker"), self.s_task_name, self.s_service_name
            )
            + py_cmd
            + ContentGenerator.get_echo_end(),
        )
