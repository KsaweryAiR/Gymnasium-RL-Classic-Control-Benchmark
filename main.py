from config import load_hyperparameters
import os

# ==========================================
# KONFIGURACJA URUCHOMIENIA
# ==========================================
OBJECT = "lunarlander1"     # Nazwa sekcji z hyperparameters.yml (np. cartpole1, flappybird1, lunarlander1)
CONTROLLER = "DQN"       # Typ kontrolera (DQN, PID, MPC)
MODE = "train"           # Tryb działania: 'train' (nauka) lub 'test' (testowanie). Dla PID/MPC może być 'run'
RENDER = False           # Czy renderować graficznie środowisko (używane głównie przy MODE = 'test')
# ==========================================

def main():
    obj_name = OBJECT.lower()
    controller_name = CONTROLLER.upper()
    mode_name = MODE.lower()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(base_dir, 'hyperparameters.yml')

    hp = load_hyperparameters(yaml_path, obj_name, controller_name)

    if controller_name == 'DQN':
        from Controllers.DQN.trainer import Trainer
        trainer = Trainer(hyperparameter_set=obj_name, hyperparameters=hp)

        if mode_name == 'train':
            trainer.train()
        elif mode_name == 'test':
            trainer.test(render=RENDER)
        else:
            print(f"Błąd: Dla DQN wymagany jest tryb 'train' lub 'test'. Podano: '{MODE}'")

    elif controller_name == 'PID':
        print(f"Uruchamiam kontroler PID dla obiektu {obj_name} w trybie {mode_name}...")

    elif controller_name == 'MPC':
        print(f"Uruchamiam kontroler MPC dla obiektu {obj_name} w trybie {mode_name}...")

    else:
        print(f"Błąd: Kontroler '{CONTROLLER}' nie jest obsługiwany.")

if __name__ == '__main__':
    main()