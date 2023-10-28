import numpy as np
import pandas as pd
from way_info import way

class ActionGenerator:
    def __init__(self):
        self.myway = way()
        self.next_node = np.arange(self.myway.n_pois)
        self.charge = np.array([0, 0.3, 0.5, 0.8])
        self.rest = np.array([0, 0.3, 0.6, 0.9, 1])

    def generate_actions(self):
        actions = []

        for node in self.next_node:
            if node in range(self.myway.n_ch):
                for ch in self.charge:
                    actions.append([node, ch, 0])
            elif node in range(self.myway.n_ch, self.myway.n_pois):
                for r in self.rest:
                    actions.append([node, 0, r])

        return actions

    def save_to_csv(self, filename):
        actions = self.generate_actions()
        df = pd.DataFrame(actions, columns=['next_node', 'charge', 'rest'])
        print(df)
        df.to_csv(filename, index=False)


generator = ActionGenerator()
generator.save_to_csv("actions.csv")



