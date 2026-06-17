import yaml

class Hyperparameters:
   
    def __init__(self, **entries):
        self.__dict__.update(entries)


def load_hyperparameters(path: str, hyperparameter_set: str, controller_name: str) -> Hyperparameters:
    with open(path, 'r') as f:
        all_sets = yaml.safe_load(f)

    if hyperparameter_set not in all_sets:
        raise KeyError(f"Błąd: Nie znaleziono obiektu '{hyperparameter_set}' w pliku {path}")


    env_params = all_sets[hyperparameter_set].get('env', {})
    
  
    controller_params = all_sets[hyperparameter_set].get(controller_name.lower(), {})


    if not controller_params:
         print(f"Ostrzeżenie: Brak dedykowanej sekcji '{controller_name.lower()}' dla obiektu '{hyperparameter_set}' w YAML.")


    combined_params = {**env_params, **controller_params}

    return Hyperparameters(**combined_params)