import numpy as np


    
class NNS:
    def __init__(self, model, layer_index, classes):
        self.classes= classes
        self.layer_index = layer_index
        self.model = model
        self.hidden_model = keras.Model(inputs=self.model.inputs, outputs=self.model.layers[layer_index].output)
        self.follow_model = tf.keras.Sequential(model.layers[layer_index+1:]) 
        self.follow_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    def fetch_internals(self, x):
        x_hidden_layer_values = []
        batch_size = 1000
        for i in range(x.shape[0]//batch_size):
            x_batch = x_[i*batch_size:(i+1)*batch_size]
            hidden_values_batch = self.hidden_model(x_batch)
            item = hidden_values_batch
            if item.ndim==4:
                item = item.numpy().reshape(item.shape[0], -1, item.shape[-1])
                item = np.average(item, axis=1)
            x_hidden_layer_values.append(item)
        
        hidden_layer_values = np.concatenate(x_hidden_layer_values) 
        self.internals = hidden_layer_values
        self.probs = self.model.predict(x)
        
    def cosine_similarity(self, vector, matrix):
        dot_product = np.dot(matrix, vector)
        vector_norm = np.linalg.norm(vector)
        matrix_norms = np.linalg.norm(matrix, axis=1)
        cosine_sim = dot_product / (vector_norm * matrix_norms)
        return cosine_sim

    def euclidean_distance(self, vector, matrix):
        diff = matrix - vector
        dist = np.sqrt(np.sum(diff**2, axis=1))
        return dist
    
    def smooth(self, x, k, a):
        org = self.hidden_model(x)
        org = org.numpy().reshape(org.shape[0], -1, org.shape[-1])
        org = np.average(org, axis=1)[0]
        cos_dis = 1-self.cosine_similarity(org, self.internals)
        index = np.argsort(cos_dis)[:k]
        probs = self.probs[index]
        x_prob = self.model.predict(x)[0]
        final_prob = a*x_prob+(1-a)*(1/k)*(np.sum(probs, axis=0))
        return final_prob



class NNS_NORM:
    def __init__(self, model, layer_index, classes):
        self.classes= classes
        self.layer_index = layer_index
        self.class_freqs = []
        self.class_ns = []
        self.class_patterns = []
        self.model = model
        self.hidden_model = keras.Model(inputs=self.model.inputs, outputs=self.model.layers[layer_index].output)
        self.follow_model = tf.keras.Sequential(model.layers[layer_index+1:]) 
        self.follow_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    def fetch_internals(self, x, y):        
        x_hidden_layer_values = []
        batch_size = 1000
        for i in range(x.shape[0]//batch_size):
            x_batch = x[i*batch_size:(i+1)*batch_size]
            hidden_values_batch = self.hidden_model(x_batch)
            item = hidden_values_batch
            if item.ndim==4:
                item = item.numpy().reshape(item.shape[0], -1, item.shape[-1])
                item = np.average(item, axis=1)
            x_hidden_layer_values.append(item)

        matrix = np.concatenate(x_hidden_layer_values) 
        matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized_matrix = matrix / matrix_norms
        
        self.internals = normalized_matrix
        self.probs = self.model.predict(x)
        
    def cosine_similarity(self, vector, matrix):
        dot_product = np.dot(matrix, vector)
        vector_norm = np.linalg.norm(vector)
        matrix_norms = np.linalg.norm(matrix, axis=1)
        cosine_sim = dot_product / (vector_norm * matrix_norms)
        return cosine_sim
    
    def euclidean_distance(self, vector, matrix):
        diff = matrix - vector
        dist = np.sqrt(np.sum(diff**2, axis=1))
        return dist
    
    def smooth(self, x, k, a):
        org = self.hidden_model(x)
        org = org.numpy().reshape(org.shape[0], -1, org.shape[-1])
        org = np.average(org, axis=1)[0]
        org_norm = np.linalg.norm(org)
        org = org / org_norm
        cos_dis = 1-self.cosine_similarity(org, self.internals)
        index = np.argsort(cos_dis)[:k]
        probs = self.probs[index]
        x_prob = self.model.predict(x)[0]
        final_prob = a*x_prob+(1-a)*(1/k)*(np.sum(probs, axis=0))
        
        return final_prob


# class Dissector:
#     def __init__(self, model, classes):
#         self.classes= classes
#         self.model = model
#         self.sub_models = []
    
#     def prepare_submodels(self, x, y, layer_indexes):
#         for i in layer_indexes:
#             sub_model = tf.keras.models.Sequential(self.model.layers[:i+1])
#             if isinstance(self.model.layers[i], tf.keras.layers.Conv2D):
#                 new_fc_layer = tf.keras.layers.Dense(classes, activation='softmax')
#                 sub_model.add(tf.keras.layers.Flatten())
#                 sub_model.add(new_fc_layer)
#             else:
#                 new_fc_layer = tf.keras.layers.Dense(classes, activation='softmax')
#                 sub_model.add(new_fc_layer)
#             sub_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#             sub_model.fit(x, y, epochs=classes, batch_size=64)
#             self.sub_models.append(sub_model)
        
#     def test(self, x):
#         x_predict_label = np.argmax(self.model(x))
#         sub_model_probs = [m(x)[0].numpy() for m in self.sub_models]
#         sub_model_labels = [np.argmax(prob) for prob in sub_model_probs]
#         sub_weights = list(range(1, len(self.sub_models)+1))
        
#         scores = []
#         for i in range(len(sub_model_labels)):
#             sub_prob, sub_label = sub_model_probs[i], sub_model_labels[i]
#             sub_probe_predict = sub_prob[x_predict_label]
#             max_probe = np.max(sub_prob)
#             if x_predict_label == sub_label:
#                 secondHighest = np.sort(sub_prob)[-2]
#                 score = sub_probe_predict/(sub_probe_predict + secondHighest + 1e-100)
#             else:
#                 score = 1.0 - max_probe/(max_probe + sub_probe_predict + 1e-100)
#             scores.append(score)
#         result = 0.0
#         for i in range(len(scores)):
#             result += scores[i]*sub_weights[i]
#         return result/sum(sub_weights)
    
    
    
class FAST:
    def __init__(self, model, layer_index, classes):
        self.classes= classes
        self.layer_index = layer_index
        self.class_freqs = []
        self.class_ns = []
        self.model = model
        self.hidden_model = keras.Model(inputs=self.model.inputs, outputs=self.model.layers[layer_index].output)
        self.follow_model = tf.keras.Sequential(model.layers[layer_index+1:]) 
        self.follow_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    def activation_mask(self, internals, strategy=None, theta=None):
        if strategy == 'xplore':
            internals_normlized = [(v-np.min(v))/(np.max(v)-np.min(v)+1e-10) for v in internals]
            activate_mask = np.array([np.where(i>theta, 1, 0) for i in internals_normlized])
        if strategy == 'max':
            ranks = np.argsort(-internals, axis=1)
            activate_mask = np.zeros(internals.shape, dtype=np.int8)
            NUMS = int(ranks.shape[1] * theta)
            for i in range(activate_mask.shape[0]):
                activate_mask[i][ranks[i][:NUMS]] = 1
        if strategy == 'mean':
            activate_mask = np.array([np.where(internals[i]>theta[i], 1, 0) for i in range(internals.shape[0])])
        return activate_mask   
    
    def fetch_freqs(self, x, y, matrix, strategy, theta):        
        x_hidden_layer_values = []
        batch_size = 100
        for i in range(x.shape[0]//batch_size):
            x_batch = x[i*batch_size:(i+1)*batch_size]
            hidden_values_batch = self.hidden_model(x_batch)
            item = hidden_values_batch
            if item.ndim==4:
                item = item.numpy().reshape(item.shape[0], -1, item.shape[-1])
                item = np.average(item, axis=1)
            x_hidden_layer_values.append(item)
        
        hidden_layer_values = np.concatenate(x_hidden_layer_values) 
        
        for c in range(self.classes):
            c_idxes = matrix[c]
            class_hidden_layer_values = hidden_layer_values[c_idxes]
            class_activation_masks = self.activation_mask(class_hidden_layer_values, strategy, theta) 
            class_freq = np.sum(class_activation_masks, axis=0)/class_activation_masks.shape[0]
            self.class_freqs.append(class_freq) 

    
    def fetch_ns(self, x, y, matrix):        
        layer_index = self.layer_index
        for c in range(self.classes):
            c_idxes = matrix[c]
            c_x = x[c_idxes]
            batch_size = 100
            hidden_layer_values = []
            
            for i in range((c_x.shape[0]//batch_size)+1):
                x_batch = c_x[i*batch_size:(i+1)*batch_size]
                hidden_values_batch = self.hidden_model(x_batch)
                hidden_layer_values.append(hidden_values_batch)
            hidden_layer_values = np.concatenate(hidden_layer_values) 
            
            ns = []
            org_conf = np.max(model.predict(feed), axis=1)
            for i in range(self.model.layers[layer_index].output_shape[-1]):
                class_pattern = np.ones(self.model.layers[layer_index].output_shape[-1])
                class_pattern[i] = 0 
                layer_outputs = hidden_layer_values  * class_pattern
                after_conf = np.max(self.follow_model.predict(layer_outputs), axis=1)
                ns.append(org_conf-after_conf)
            self.class_ns.append(np.array(ns))
            

    def fetch_class_patterns_freqs(self, p):
        self.class_patterns = []
        for c in range(self.classes):
            class_freq = self.class_freqs[c]
            class_pattern = np.zeros(class_freq.shape, dtype=np.int8)
            NUMS = int(class_freq.shape[0] * p)
            ranks = np.argsort(-class_freq)
            class_pattern[ranks[:NUMS]] = 1
            self.class_patterns.append(class_pattern) 
    
    
    def fetch_class_patterns_ns(self, p):
        self.class_patterns = []
        for c in range(self.classes):
            class_freq = self.class_ns[c]
            class_pattern = np.zeros(class_freq.shape, dtype=np.int8)
            NUMS = int(class_freq.shape[0] * p)
            ranks = np.argsort(-class_freq)
            class_pattern[ranks[:NUMS]] = 1
            self.class_patterns.append(class_pattern) 
        
    
    def class_acc_pattern(self, x, y, matrix, c):
        labels = np.argmax(y, axis=1)
        c_idxes = matrix[c]
        c_x = x[c_idxes]
        class_pattern = self.class_patterns[c]
        batch_size = 100
        
        hidden_layer_values = []
        for i in range((c_x.shape[0]//batch_size)+1):
            x_batch = c_x[i*batch_size:(i+1)*batch_size]
            hidden_values_batch = self.hidden_model(x_batch)
            hidden_layer_values.append(hidden_values_batch)
            
        hidden_layer_values = np.concatenate(hidden_layer_values) 
        layer_outputs = hidden_layer_values  * class_pattern
        _loss, _acc = self.follow_model.evaluate(layer_outputs, y[c_idxes], verbose=0)   
        return _acc 
    
    
    def test(self, x):
        x_predict_label = np.argmax(self.model(x))
        x_hidden_values = self.hidden_model(x)
        c = x_predict_label
        class_pattern = self.class_patterns[c]
        layer_outputs = x_hidden_values * class_pattern
        return self.follow_model(layer_outputs)[0].numpy()
    

    
    