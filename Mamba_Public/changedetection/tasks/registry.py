from importlib import import_module


TRAINERS = {
    "bda": ("changedetection.tasks.bda", "BDATrainer"),
    "bright": ("changedetection.tasks.bright", "BRIGHTTrainer"),
}

INFERERS = {
    "bda": ("changedetection.tasks.bda", "BDAInferer"),
}


def _load_entry(entry):
    module_name, attr_name = entry
    module = import_module(module_name)
    return getattr(module, attr_name)


def get_trainer(task_name):
    return _load_entry(TRAINERS[task_name])


def get_inferer(task_name):
    return _load_entry(INFERERS[task_name])
