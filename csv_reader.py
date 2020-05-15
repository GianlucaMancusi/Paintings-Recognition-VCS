import pandas as pd

class InfoTable():

    def __init__(self, filename):
        self.data = pd.read_csv(filename, header=0)
    
    def painting_info(self, code):
        filename = f'{code:03d}.png'
        d = self.data.loc[self.data['Image'] == filename].to_dict(orient='records')
        return d[0] if len(d) > 0 else None
    
    def room_of(self, code):
        info = self.painting_info(code)
        return info['Room'] if not info is None else None

if __name__ == "__main__":
    table = InfoTable('dataset/data.csv')
    print(table.room_of(94))