import base64

def convert_logo():
    with open("ScottieDogHead.png", "rb") as img_file:
        base64_string = base64.b64encode(img_file.read()).decode()
        print("Your base64 string:")
        print(base64_string)

convert_logo()