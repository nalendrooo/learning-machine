import gdown

url = "https://drive.google.com/drive/folders/1RpFq6HzvINZnmIT1Yr4hFFWre41yU4Z9"
output = "model.h5"
gdown.download(url, output, quiet=False, fuzzy=True)
