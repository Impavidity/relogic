import importlib


def get_model(model):
    Model = importlib.import_module('{}'.format(model)).Model
    return Model


def get_constructor(constructor):
    Constructor = importlib.import_module('{}'.format(constructor)).Constructor
    return Constructor
    # return LocalDatasetModuleFactoryWithScript(
    #     combined_path, download_mode=download_mode, dynamic_modules_path=dynamic_modules_path
    # ).get_module()


def get_evaluator(evaluate_tool):
    EvaluateTool = importlib.import_module('{}'.format(evaluate_tool)).EvaluateTool
    return EvaluateTool
