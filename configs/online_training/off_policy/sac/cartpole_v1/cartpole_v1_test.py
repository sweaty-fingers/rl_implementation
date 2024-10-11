from configs.online_training.off_policy.shared import CONFIG

root_data_dir = CONFIG["dirs"]["root"]["dataset"]

training_info = {"processed_dataname": "cartpole_v1_test"}
networks = {
    "actor": {
        "class_name": "DiscreteFCNNPolicyNetwork"
    },
    "critic_1":{
        "class_name": "FCNNQNetwork"
    },
    "critic_2":{
        "class_name": "FCNNQNetwork"
    }
}
CONFIG["networks"] = networks
