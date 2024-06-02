import pandas as pd
df = pd.read_csv('Datasets/converted_dataset.csv')

for i in range(0, len(df)):
    model = df.iloc[i]['model']
    price_eur = df.iloc[i]['Price_Euro']
    processor = df.iloc[i]['processor']
    ram = df.iloc[i]['ram']
    battery = df.iloc[i]['battery']
    display = df.iloc[i]['display']
    camera = df.iloc[i]['camera']
    os = df.iloc[i]['os']
    
    #wrote in file
    with open('output.txt', 'a', encoding='utf-8') as f:
        f.write("Model: "+model+"\n")
        f.write("Price_Euro: "+str(price_eur)+"\n")
        f.write("Processor: "+processor+"\n")
        f.write("RAM: "+str(ram)+"\n")
        f.write("Battery: "+str(battery)+"\n")
        f.write("Display: "+str(display)+"\n")
        f.write("Camera: "+str(camera)+"\n")
        f.write("OS: "+str(os)+"\n")
        f.write("\n")