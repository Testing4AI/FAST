import numpy as np


class NAC:
    def __init__(self, model, layer_indexes, classes):
        self.classes= classes
        self.layer_indexes = layer_indexes
        self.model = model
        self.hidden_model = keras.Model(inputs=self.model.inputs, outputs=[self.model.layers[i].output for i in layer_indexes])
    
    def test(self, x):
        x_hidden_values = self.hidden_model(x)
        ob_counts = 0
        for layer, item in enumerate(x_hidden_values):
            if item.ndim==4:
                item = item.numpy().reshape(item.shape[0], -1, item.shape[-1])
                item = np.average(item, axis=1)
            else:
                item = item.numpy()
            v = item[0]
            layer_outputs = (v-np.min(v))/(np.max(v)-np.min(v)+1e-6)
            ob_counts = ob_counts + np.sum(layer_outputs>0.5)
        return ob_counts
    
    
    

class OBSAN:
    def __init__(self, model, layer_indexes, classes):
        self.classes= classes
        self.layer_indexes = layer_indexes
        self.layer_max_bounds = [[] for i in range(len(layer_indexes))]
        self.layer_min_bounds = [[] for i in range(len(layer_indexes))]
        self.model = model
        self.hidden_model = keras.Model(inputs=self.model.inputs, outputs=[self.model.layers[i].output for i in layer_indexes])
    
    def fetch_ranges(self, x, y, matrix):
        x_hidden_layer_values = [[] for i in range(len(self.layer_indexes))]
        batch_size = 1000
        for i in range(x.shape[0]//batch_size):
            x_batch = x[i*batch_size:(i+1)*batch_size]
            hidden_values_batch = self.hidden_model(x_batch)
            hidden_values = []
            for layer, item in enumerate(hidden_values_batch):
                if item.ndim==4:
                    item = item.numpy().reshape(item.shape[0], -1, item.shape[-1])
                    item = np.average(item, axis=1)
                x_hidden_layer_values[layer].append(item)
        
        hidden_layer_values = [np.concatenate(item) for item in x_hidden_layer_values]
        self.hidden_layer_values = hidden_layer_values
        
        for layer in range(len(self.layer_indexes)):
            for c in range(self.classes):
                c_idxes = matrix[c]
                class_hidden_layer_values = self.hidden_layer_values[layer][c_idxes] 
                max_bounds = np.max(class_hidden_layer_values, axis=0) 
                min_bounds = np.min(class_hidden_layer_values, axis=0)
                self.layer_max_bounds[layer].append(max_bounds)
                self.layer_min_bounds[layer].append(min_bounds)
            
    def test(self, x):
        x_predict_label = np.argmax(self.model(x))
        x_hidden_values = self.hidden_model(x)
        ob_counts = 0
        for layer, item in enumerate(x_hidden_values):
            if item.ndim==4:
                item = item.numpy().reshape(item.shape[0], -1, item.shape[-1])
                item = np.average(item, axis=1)
            else:
                item = item.numpy()
            c = x_predict_label
            class_max_bounds = self.layer_max_bounds[layer][c]
            class_min_bounds = self.layer_min_bounds[layer][c]
            layer_outputs = item[0]
            ob_counts = ob_counts + np.sum(layer_outputs>class_max_bounds) + np.sum(layer_outputs<class_min_bounds)
        return ob_counts
    
    
    
    