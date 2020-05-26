#imports
import numpy as np
import pandas as pd
import math

class RProp:
    
    def __init__(self, conf, delta = 0.1, d_min = 10**(-6), d_max = 50, eta_add = 1.2, eta_min = 0.5):
       self.input  = np.empty(conf[0])
       self.hidden = np.empty(conf[1])
       self.output = np.empty(conf[2])
       self.bias_h = np.ones(conf[1])
       self.bias_o = np.ones(conf[2])
       self.weights_ih = np.random.rand(conf[0], conf[1])
       self.weights_ho = np.random.rand(conf[1], conf[2])
       self.delta = delta
       self.d_min = d_min
       self.d_max = d_max
       self.eta_add = eta_add
       self.eta_min = eta_min

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
        # return (np.exp(2*x)-1) / (np.exp(2*x)+1)
    
    def sig_derivative(self, x):
        return self.sigmoid(x) * (1-self.sigmoid(x))
        # return 1-self.sigmoid(x)**2
    
    def mae(self, pred, target):
        return np.sum(np.abs(pred-target)) / len(target)
    
    def mae_derivative(self, x, y):
        if x > y: z = 1
        if x < y: z = -1
        return z

    def train(self, x, y, max_epoch = 100):
        epoch = 1
        error = []
        all_error=[]
        prev_ho = np.vstack((np.zeros(self.output.shape), np.zeros(self.weights_ho.shape)))
        prev_ih = np.vstack((np.zeros(self.hidden.shape), np.zeros(self.weights_ih.shape)))
        prev_delta_ho = np.full(prev_ho.shape, self.delta)
        prev_delta_ih = np.full(prev_ih.shape, self.delta)
        prev_delta_who = np.zeros(prev_ho.shape)
        prev_delta_wih = np.zeros(prev_ih.shape) 


        while epoch <= max_epoch:
            acc_ho = np.vstack((np.zeros(self.output.shape), np.zeros(self.weights_ho.shape)))
            acc_ih = np.vstack((np.zeros(self.hidden.shape), np.zeros(self.weights_ih.shape)))
            
            for x_row, y_row in zip(x, y):
                
                #feedforward
                self.input = x_row
                self.hidden = np.vectorize(self.sigmoid)(np.einsum('i,ij->j', self.input, self.weights_ih))
                self.output = np.vectorize(self.sigmoid)(np.einsum('i,ij->j', self.hidden, self.weights_ho))
                error.append(self.mae(self.output, y_row))

                #backprop: output-hidden
                term2 = np.vectorize(self.sig_derivative)(np.einsum('i,ij->j', self.hidden, self.weights_ho))
                term3 = np.vectorize(self.mae_derivative)(self.output, y_row)
                grad_ho = np.einsum('i,j->ij', self.hidden, term2 * term3)
                grad_bo = term2 * term3
                grad_ho = np.vstack((grad_ho, grad_bo))

                #backprop: hidden-input
                term3 = np.dot(term2 * term3, self.weights_ho.T)
                term2 = np.vectorize(self.sig_derivative)(np.einsum('i,ij->j', self.input, self.weights_ih))
                grad_ih = np.einsum('i,j->ij', self.input, term2 * term3)
                grad_bh = term2 * term3
                grad_ih = np.vstack((grad_ih, grad_bh))
                
                #accumulate derivatives
                acc_ho += grad_ho
                acc_ih += grad_ih
            
            #update-value                
            delta_ho = np.zeros(acc_ho.shape)
            delta_ih = np.zeros(acc_ih.shape)
            delta_who = np.zeros(acc_ho.shape)
            delta_wih = np.zeros(acc_ih.shape)

            #Case 1: no sign changed
            #update hidden-output weights
            for i,j in zip(*np.where(acc_ho * prev_ho > 0)):
                delta_ho[i][j] = np.minimum(self.d_max, acc_ho * self.eta_add)[i][j]
                delta_who[i][j] = delta_ho[i][j] * np.sign(acc_ho)[i][j] * -1
                if i==acc_ho.shape[0]-1: self.bias_o += delta_who[i][j]
                else: self.weights_ho += delta_who[i][j]
                
            #update input-hidden weights
            for i,j in zip(*np.where(acc_ih * prev_ih > 0)):
                delta_ih[i][j] = np.minimum(self.d_max, acc_ih * self.eta_add)[i][j]
                delta_wih[i][j] = delta_ih[i][j] * np.sign(acc_ih)[i][j] * -1
                if i==acc_ih.shape[0]-1: self.bias_h += delta_wih[i][j]
                else: self.weights_ih += delta_wih[i][j]
            
            #Case 2: equal zero
            #update hidden-output weights
            for i,j in zip(*np.where(acc_ho * prev_ho == 0)):
                delta_ho[i][j] = prev_delta_ho[i][j]
                delta_who[i][j] = delta_ho[i][j] * np.sign(acc_ho)[i][j] * -1
                if i==acc_ho.shape[0]-1: self.bias_o += delta_who[i][j]
                else: self.weights_ho += delta_who[i][j]

            #update input-hidden weights
            for i,j in zip(*np.where(acc_ih * prev_ih == 0)):
                delta_ih[i][j] = prev_delta_ih[i][j]
                delta_wih[i][j] = delta_ih[i][j] * np.sign(acc_ih)[i][j] * -1
                if i==acc_ih.shape[0]-1: self.bias_h += delta_wih[i][j]
                else: self.weights_ih += delta_wih[i][j]

            #Case 3: sign changed
            #update hidden-output weights
            for i,j in zip(*np.where(acc_ho * prev_ho < 0)):
                delta_ho[i][j] = np.maximum(self.d_min, acc_ho * self.eta_min)[i][j]
                if i==acc_ho.shape[0]-1: self.bias_o -= prev_delta_who[i][j]
                else: self.weights_ho -= prev_delta_who[i][j]
                acc_ho[i][j] = 0
            
            #update input-hidden weights
            for i,j in zip(*np.where(acc_ih * prev_ih < 0)):
                delta_ih[i][j] = np.maximum(self.d_min, acc_ih * self.eta_min)[i][j]
                if i==acc_ih.shape[0]-1: self.bias_h -= prev_delta_wih[i][j]
                else: self.weights_ih -= prev_delta_wih[i][j]
                acc_ih[i][j] = 0
            
            #save previous data
            prev_ho = acc_ho
            prev_ih = acc_ih
            prev_delta_ho = delta_ho
            prev_delta_ih = delta_ih
            prev_delta_who = delta_who
            prev_delta_wih = delta_wih
            
            epoch += 1
            all_error.append(np.average(error))
        # all_error = pd.DataFrame(all_error)
        # all_error.to_csv('errors.csv')
    
    def test(self, x):
        pred=[]
        for x_row in x:
                #feedforward
                self.input = x_row
                self.hidden = np.vectorize(self.sigmoid)(np.einsum('i,ij->j', self.input, self.weights_ih))
                self.output = np.vectorize(self.sigmoid)(np.einsum('i,ij->j', self.hidden, self.weights_ho))
                pred.append(self.output)
        return np.array(pred)
    
    def accuracy(self, pred, y):
        return 1-np.average([self.mae(pred_row, y_row) for pred_row in pred for y_row in y])