from models import load_model, MODELS_PATH


def hex_to_c_array(hex_data, var_name):
    # convert tflite model to c array

    c_str = ""

    # Create header guard
    c_str += "#ifndef " + var_name.upper() + "_H\n"
    c_str += "#define " + var_name.upper() + "_H\n\n"

    # Add array length at top of file
    c_str += (
        "\nstatic const unsigned int "
        + var_name
        + "_len = "
        + str(len(hex_data))
        + ";\n"
    )

    # Declare C variable
    c_str += "static const unsigned char " + var_name + "[] = {"
    hex_array = []
    for i, val in enumerate(hex_data):

        # Construct string from hex
        hex_str = format(val, "#04x")

        # Add formatting so each line stays within 80 characters
        if (i + 1) < len(hex_data):
            hex_str += ","
        if (i + 1) % 12 == 0:
            hex_str += "\n "
        hex_array.append(hex_str)

    # Add closing brace
    c_str += "\n " + format(" ".join(hex_array)) + "\n};\n\n"

    # Close out header guard
    c_str += "#endif //" + var_name.upper() + "_H"

    return c_str


def convert_tflite_to_c(cfg, train_type):
    # load and convert (trained) tflite model to a c .h file
    with open(MODELS_PATH + cfg["path"] + train_type + ".tflite", "rb") as f:
        tflite_model_content = f.read()

    # convert to c file
    c_model_name = "".join(cfg["path"].split("_")) + train_type
    with open(MODELS_PATH + "/cfiles/" + c_model_name + ".h", "w") as file:
        file.write(hex_to_c_array(tflite_model_content, c_model_name))


def save_h5_model(cfg, train_type, mct):
    # load and save (trained) tensorflow model
    model = load_model(cfg, train_type, mct)
    model.save(MODELS_PATH + cfg["path"] + train_type + ".h5")
    print(".h5 file saved successfully")


def save_h5_model(cfg, train_type, mct):
    # load and save (trained) tensorflow model (.h5: legacy format)
    model = load_model(cfg, train_type, mct)
    model.save(MODELS_PATH + cfg["path"] + mct + train_type + ".h5")
    print(".h5 file saved successfully")
