"""Family tree utility functions."""
import pandas as pd
from typing import Dict

def calculate_family_depths(df: pd.DataFrame) -> Dict[str, int]:
    """Calculate family depth for each model."""
    depths = {}
    computing = set()
    
    def get_depth(model_id: str) -> int:
        if model_id in depths:
            return depths[model_id]
        if model_id in computing:
            depths[model_id] = 0
            return 0
        
        computing.add(model_id)
        
        try:
            if df.index.name == 'model_id':
                row = df.loc[model_id]
            else:
                rows = df[df.get('model_id', '') == model_id]
                if len(rows) == 0:
                    depths[model_id] = 0
                    computing.remove(model_id)
                    return 0
                row = rows.iloc[0]
            
            parent_id = row.get('parent_model')
            if parent_id and pd.notna(parent_id):
                parent_str = str(parent_id)
                if parent_str != 'nan' and parent_str != '':
                    if df.index.name == 'model_id' and parent_str in df.index:
                        depth = get_depth(parent_str) + 1
                    elif df.index.name != 'model_id':
                        parent_rows = df[df.get('model_id', '') == parent_str]
                        if len(parent_rows) > 0:
                            depth = get_depth(parent_str) + 1
                        else:
                            depth = 0
                    else:
                        depth = 0
                else:
                    depth = 0
            else:
                depth = 0
        except (KeyError, IndexError):
            depth = 0
        
        depths[model_id] = depth
        computing.remove(model_id)
        return depth
    
    if df.index.name == 'model_id':
        for model_id in df.index:
            if model_id not in depths:
                get_depth(str(model_id))
    else:
        for _, row in df.iterrows():
            model_id = str(row.get('model_id', ''))
            if model_id and model_id not in depths:
                get_depth(model_id)
    
    return depths

