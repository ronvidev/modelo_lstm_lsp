import os
import argparse
import wifi


def get_networks():
    redes_disponibles = wifi.Cell.all('wlan0')
    print("Redes disponibles:")
    for i, red in enumerate(redes_disponibles):
        print(f"{i}: SSID: {red.ssid}, Signal: {red.signal}")
    

# def connect_network():
#     indice_red = int(input("Ingresa el índice de la red a la que deseas conectarte: "))
#     red_seleccionada = redes_disponibles[indice_red]

#     contraseña = input("Ingresa la contraseña de la red: ")

#     es_red_abierta = False if red_seleccionada.encryption_type != 'open' else True
#     if es_red_abierta:
#         wifi.Scheme.for_cell('wlan0', red_seleccionada.ssid, red_seleccionada, '')
#         print(f"Conectado a la red {red_seleccionada.ssid}.")
#     else:
#         wifi.Scheme.for_cell('wlan0', red_seleccionada.ssid, red_seleccionada, contraseña)
#         print(f"Conectado a la red {red_seleccionada.ssid} con contraseña.")


def turn_on():
    os.system("sudo ifconfig wlan0 up")
    print("WiFi encendido.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interacción con el Wifi (Ubuntu 20.04)",
        epilog="Sólo está probado en Ubuntu 20.04 y versiones de Python 3.8.10 y 3.9.0",
        usage="wifi_actions.py <function> [options]",
    )

    # Argumentos posicionales (obligatorios)
    parser.add_argument(
        "function",
        type=str,
        help="[turn_on] Encender el Wifi | [get_networks] Retorna una lista de redes disponibles | [connect_network] Conectar a una red (requiere --ssid y --password)",
    )

    # Argumentos opcionales
    parser.add_argument("--ssid", type=str, help="Nombre de la red Wifi")
    parser.add_argument("--password", type=str, help="Contraseña de la red Wifi")

    args = parser.parse_args()

    print("Función en uso:", args.function)
    
    functions = {
        "turn_on": turn_on,
        "get_networks": get_networks,
        # "connect_network": connect_network,
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
