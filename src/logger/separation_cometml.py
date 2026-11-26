from datetime import datetime

from src.logger.cometml import CometMLWriter
from src.utils.io_utils import ROOT_PATH


class SeparationCometMLWriter(CometMLWriter):
    def __init__(
        self,
        logger,
        project_config,
        project_name,
        workspace=None,
        run_id=None,
        run_name=None,
        mode="online",
        offline_directory=".cometml-runs",
        **kwargs,
    ):
        import comet_ml

        self.run_id = run_id
        experiment_args = {
            "log_code": False,
            "log_graph": False,
            "auto_metric_logging": False,
            "auto_param_logging": False,
            "auto_output_logging": None,
            "log_env_details": False,
            "log_git_metadata": False,
        }
        if mode == "offline":
            experiment_args["offline_directory"] = str(ROOT_PATH / offline_directory)

        if project_config["trainer"].get("resume_from") is not None:
            experiment_class = (
                comet_ml.ExistingOfflineExperiment
                if mode == "offline"
                else comet_ml.ExistingExperiment
            )
            self.exp = experiment_class(previous_experiment=run_id, **experiment_args)
        else:
            experiment_class = (
                comet_ml.OfflineExperiment if mode == "offline" else comet_ml.Experiment
            )
            self.exp = experiment_class(
                project_name=project_name,
                workspace=workspace,
                experiment_key=run_id,
                **experiment_args,
            )

        self.exp.set_name(run_name)
        self.exp.log_parameters(project_config)
        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

    def close(self):
        self.exp.end()
