import os
import re
import subprocess
from models import MODELS_PATH


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


def install_java(package: str = "openjdk-17-jdk", version: int = 17) -> bool:
    """Checks for a Java installation and installs it if necessary"""
    try:
        result = subprocess.run(
            ["java", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        version_output = result.stdout.splitlines()[0]
        match = re.search(
            r"(\d+)\.(\d+)\.(\d+)", version_output
        )  # Match version in form major.minor.patch
        print(f"Found Java version: {match.group(0)}")
        if match:
            major_version = int(match.group(1))
            if major_version == version:
                return True
            else:
                print(f"Java {version} is not installed. Installing correct version...")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Java not installed. Installing...")

    try:
        is_root = os.geteuid() == 0
        prefix = [] if is_root else ["sudo"]
        with open(os.devnull, "w") as devnull:
            subprocess.run(
                prefix + ["apt", "install", "-y", package],
                check=True,
                stdout=devnull,
                stderr=devnull,
            )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Installation error: {e}")
        return False


def mct_setup():
    # mct converter requires Java
    if install_java():
        print(f"Java installed")
    else:
        print(f"Java missing and installation failed")
