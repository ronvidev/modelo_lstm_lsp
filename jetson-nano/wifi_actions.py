import os
import argparse
import subprocess


def get_networks():
    try:
        resultado = subprocess.run(['nmcli', '-f', 'SSID,SIGNAL', 'dev', 'wifi', 'list'], capture_output=True, text=True, check=True)
        lineas = resultado.stdout.split('\n')
        redes_wifi = []
        for linea in lineas:
            if linea.strip() and 'SSID' not in linea:  # Ignorar líneas en blanco y el encabezado
                ssid, potencia = linea.strip()[:-2].strip(), linea.strip()[-2:]
                if ssid not in [par[0] for par in redes_wifi]:
                    redes_wifi.append((ssid, int(potencia)))  # Convertir la potencia a entero y añadir a la lista
        print(redes_wifi)
        return redes_wifi
    except subprocess.CalledProcessError as e:
        print("Error al intentar obtener la lista de WiFi disponible:", e)
        return []
    

def connect_network(ssid, password):
    try:
        subprocess.run(['nmcli', 'dev', 'wifi', 'connect', ssid, 'password', password], check=True)
        print(f"Conectado a la red Wi-Fi '{ssid}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error al intentar conectar a la red Wi-Fi '{ssid}':", e)


def turn_on():
    os.system("nmcli radio wifi on")
    print("WiFi encendido.")

def turn_off():
    os.system("nmcli radio wifi off")
    print("WiFi apagado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interacción con el Wifi (Ubuntu 20.04)",
        epilog="Sólo está probado en Ubuntu 20.04 y versiones de Python 3.8.10 y 3.9.0",
        usage="wifi_actions.py <function> [args optionals]",
    )

    # Argumentos posicionales (obligatorios)
    parser.add_argument(
        "function",
        type=str,
        help="[turn_on] Encender el Wifi | [turn_off] Apagar el Wifi | [get_networks] Retorna una lista de redes disponibles | [connect_network] Conectar a una red (requiere --ssid y --password)",
    )

    # Argumentos opcionales
    parser.add_argument("--ssid", type=str, help="Nombre de la red Wifi")
    parser.add_argument("--password", type=str, help="Contraseña de la red Wifi")

    args = parser.parse_args()

    print("Función en uso:", args.function)
    
    functions = {
        "turn_on": turn_on,
        "turn_off": turn_off,
        "get_networks": get_networks,
        "connect_network": connect_network,
    }
    
    function = functions[args.function]
    
    if args.function == "connect_network":
        ssid = args.ssid
        password = args.password
        
        if None not in [ssid, password]:
            function(ssid, password)
        else:
            print("Faltan credenciales")
    else:
        function()
