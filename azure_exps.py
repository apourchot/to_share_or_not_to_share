import configparser

from azureml.core import ContainerRegistry, Datastore, Experiment, Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.data.data_reference import DataReference
from azureml.train.dnn import PyTorch


def launch_azure_exp(**kwargs):

    # read azureml config file
    config = configparser.ConfigParser()
    config.read("config.ini")

    # azureml account stuff
    SUBSCRIPTION_ID = config['azureml']['SUBSCRIPTION_ID']
    RESOURCE_GROUP = config['azureml']['RESOURCE_GROUP']
    WORKSPACE_NAME = config['azureml']['WORKSPACE_NAME']
    WORKSPACE_REGION = config['azureml']['WORKSPACE_REGION']

    # docker setup
    DOCKER_IMAGE = "ai-research:v1.0"
    DOCKER_CONTAINER_REGISTRY_ADDRESS = config['azureml']['DOCKER_CONTAINER_REGISTRY_ADDRESS']
    DOCKER_CONTAINER_REGISTRY_USERNAME = config['azureml']['DOCKER_CONTAINER_REGISTRY_USERNAME']
    DOCKER_CONTAINER_REGISTRY_PASSWORD = config['azureml']['DOCKER_CONTAINER_REGISTRY_PASSWORD']

    # init workspace
    try:
        ws = Workspace(subscription_id=SUBSCRIPTION_ID, resource_group=RESOURCE_GROUP, workspace_name=WORKSPACE_NAME)
        ws.write_config()

    # if not found create one
    except Exception as e:
        print("Workspace not accessible. Creating new workspace below")
        ws = Workspace.create(name=WORKSPACE_NAME, subscription_id=SUBSCRIPTION_ID, resource_group=RESOURCE_GROUP,
                              location=WORKSPACE_REGION, create_resource_group=True, exist_ok=True)
        ws.get_details()
        ws.write_config()

    # init datastore
    ds = Datastore.get(workspace=ws, datastore_name='blob01')
    dr = DataReference(
        datastore=ds,
        path_on_datastore='data',
        mode='mount'
    )
    kwargs["script_params"]["--data_dir"] = dr

    # create cluster
    compute_config = AmlCompute.provisioning_configuration(vm_size=kwargs["vm_size"], vm_priority='lowpriority',
                                                           max_nodes=1)
    compute_target = ComputeTarget.create(ws, kwargs["cluster_name"], compute_config)
    compute_target.wait_for_completion(show_output=True)
    print(kwargs["cluster_name"], kwargs["vm_size"])

    # pytorch estimator
    custom_cr = ContainerRegistry()
    custom_cr.address = DOCKER_CONTAINER_REGISTRY_ADDRESS
    custom_cr.username = DOCKER_CONTAINER_REGISTRY_USERNAME
    custom_cr.password = DOCKER_CONTAINER_REGISTRY_PASSWORD
    estimator = PyTorch(source_directory=".",
                        compute_target=compute_target,
                        entry_script=kwargs["entry_script"],
                        script_params=kwargs["script_params"],
                        use_gpu=True,
                        use_docker=True,
                        custom_docker_image=DOCKER_IMAGE,
                        image_registry_details=custom_cr,
                        shm_size="50g",
                        user_managed=True,
                        environment_variables={"CUDA_VISIBLE_DEVICES": "0"}
                        )

    # create and run exp
    experiment = Experiment(ws, name=kwargs["experiment_name"])
    run = experiment.submit(estimator)
    print(run)
