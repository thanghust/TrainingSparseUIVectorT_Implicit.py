import os
import time
import tensorflow as tf
import _pickle as pickle
import numpy as np
import tensorflow.compat.v1 as v1
import pandas as pd 
import matplotlib.pyplot as plt
from model import Model
from w_initialization import xavier_weight_init
from utils.general_utils import Progbar
from DataProcessing import getMeanDay, getMeanDaybyUser

from sklearn.utils import shuffle


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    # n_items = 36
    # n_users = 38
    n_mean = 0
    n_f = 5
    lamb2 = 25
    lamb3 = 10
    lamb4 = 0.02 #0.02
    lamb5 = 0.01 #10
    lamb6 = 0.5 #15
    lamb7 = 0.1
    lamb8 = 0.1
    lamb9 = 10
    beta = 0.4
    n_epochs = 30
    lr = 0.001 #0.005
    item_bin_size = 60
    batch_size = 32
    weight_filename = "data\\weights\\recommanding_with_IU_time_drifting.weight"
    device="/cpu:0"
    maxday_cat_code = 4096
    
class RecommandationModel(Model):
    """
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        """
        self.user_placeholder = v1.placeholder(tf.int32, [None, ], name="user")
        self.item_placeholder = v1.placeholder(tf.int32, [None, ], name="item")
        self.rank_placeholder = v1.placeholder(tf.float32, [None, ], name="rank")
        self.tbin_placeholder = v1.placeholder(tf.int32, [None, ], name="tbin")
        self.tday_placeholder = v1.placeholder(tf.int32, [None, ], name="tday")
        self.mean_ud_placeholder = v1.placeholder(tf.float32, [None,], name = "mean_day_by_user")
        
        self.global_mean_placeholder = v1.placeholder(tf.float32, shape=(), name = "global_mean_rank")
        self.maxday_cat_placeholder = v1.placeholder(tf.int32, shape=[None,], name = "maxday_cat_code")


    def create_feed_dict(self, input_df, mean_rank, mean_u_day): #label_indices, label_data, label_shape, input_rank_data, input_tb_data, input_td_data, mean_rank, mean_u_day):
        feed_dict = {}

        feed_dict[self.user_placeholder] = input_df["userID"].values.astype(int)
        feed_dict[self.item_placeholder] = input_df["itemID"].values.astype(int)
        feed_dict[self.rank_placeholder] = input_df["overall"].values
        feed_dict[self.tbin_placeholder] = input_df["ITBin"].values.astype(int)
        feed_dict[self.tday_placeholder] = input_df["ReviewDay"].values.astype(int)
        feed_dict[self.global_mean_placeholder] = mean_rank
        feed_dict[self.mean_ud_placeholder] = mean_u_day.astype(float)
        feed_dict[self.maxday_cat_placeholder] = input_df["TDayCat"].values.astype(int) #max_c_day     
        return feed_dict

    def add_prediction_op(self):
        with tf.device("/cpu:0"):
            weight_initializer = xavier_weight_init()
            self.WPI = v1.get_variable("item_vector", shape = [self.config.n_items, self.config.n_f], 
                                   initializer = weight_initializer)  #I*F
            self.WPU = v1.get_variable("user_vector", shape = [self.config.n_users, self.config.n_f], 
                                   initializer =  weight_initializer)  #U*F
            self.BU = v1.get_variable("bias_user", shape= [self.config.n_users,], initializer = tf.zeros_initializer())  #U
            self.BI = v1.get_variable("bias_item", shape = [self.config.n_items,], initializer =tf.zeros_initializer()) 

            self.WBIT = v1.get_variable("bias_item_bin", shape = [self.config.n_items, self.config.item_bin_size],
                                    initializer = tf.zeros_initializer())
            self.Alpha = v1.get_variable("dev_weight", shape = [self.config.n_users,], initializer = weight_initializer)
            self.AlphaUK = v1.get_variable("dev_weightUK", shape = [self.config.n_users, self.config.n_f], initializer = weight_initializer) 
            self.WPUKT = v1.get_variable("pkut_vector", shape = [self.config.maxday_cat_code+1, self.config.n_f], 
                                   initializer =  weight_initializer)
            
            self.BTDay = v1.get_variable("day_cat_code", shape = [self.config.maxday_cat_code+1], initializer = tf.zeros_initializer())
            
            self.BCU = v1.get_variable("c_u", shape = [self.config.n_users], initializer = tf.zeros_initializer())
            self.WCU = v1.get_variable("c_u_t", shape = [self.config.maxday_cat_code+1], initializer = weight_initializer)
            
            #get the time bin values for each row 
            self.bias_item_binvalue = tf.gather_nd(self.WBIT, tf.stack([self.item_placeholder, self.tbin_placeholder], axis=1), name = "item_time_bin_value") 
        
            bias_user = tf.nn.embedding_lookup(self.BU, self.user_placeholder, name = "bias_user") 
            bias_item = tf.nn.embedding_lookup(self.BI, self.item_placeholder, name = "bias_item")  
            mean_tday = tf.nn.embedding_lookup(self.mean_ud_placeholder, self.user_placeholder) #mean tday by user t(mean)
            alpha_value = tf.nn.embedding_lookup(self.Alpha, self.user_placeholder)
            alpha_uk_value = tf.nn.embedding_lookup(self.AlphaUK, self.user_placeholder)
            
            user_vector = tf.nn.embedding_lookup(self.WPU, self.user_placeholder)
            item_vector = tf.nn.embedding_lookup(self.WPI, self.item_placeholder)
            butday = tf.nn.embedding_lookup(self.BTDay, self.maxday_cat_placeholder)
            cu_b = tf.nn.embedding_lookup(self.BCU, self.user_placeholder, name = "cu_b") 
            cu_t = tf.nn.embedding_lookup(self.WCU, self.maxday_cat_placeholder, name = "cu_t")
            pkut = tf.nn.embedding_lookup(self.WPUKT, self.maxday_cat_placeholder, name = "pkut")
         
        with tf.device(self.config.device):             
            tday_diff = tf.subtract(tf.cast(self.tday_placeholder,tf.float32), mean_tday) # t - t(mean)            
            dev_t = tf.multiply(tf.sign(tday_diff), tf.pow(tf.abs(tday_diff), self.config.beta)) #sign(t - t(mean))*abolute(t - t(mean))**beta       
            self.bias_user_tvalue =  tf.multiply(alpha_value, dev_t)
            bias_user_time = tf.add(bias_user, self.bias_user_tvalue)
            bias_user_time = tf.add(bias_user_time, butday)
            bias_item_time = tf.add(bias_item, self.bias_item_binvalue)
            cui = tf.add(cu_b, cu_t)
            bias_item_time = tf.multiply(bias_item_time, cui)
            
            self.vector_user_tvalue = tf.transpose(tf.multiply(tf.transpose(alpha_uk_value), dev_t))
            user_vector_t = tf.add(user_vector, self.vector_user_tvalue)
            user_vector_t = tf.add(user_vector_t, pkut)
            
            bias_vector = tf.reduce_sum(tf.multiply(user_vector_t, item_vector), 1)
        
            pred = tf.add(self.global_mean_placeholder, bias_user_time)
            pred = tf.add(pred, bias_item_time)
            pred = tf.add(pred, bias_vector)
        
            self.test_pred = tf.add(self.global_mean_placeholder, bias_user)
            self.test_pred = tf.add(self.test_pred, bias_item)
            self.test_pred = tf.add(self.test_pred, bias_vector)
        
        return pred

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (users, items) containing the prediction of ranks
        Returns:
            loss: A 0-d tensor (scalar)
        """   
        with tf.device(self.config.device):
            loss =  tf.nn.l2_loss(tf.subtract(pred ,self.rank_placeholder)) + 0.5 * self.config.lamb4*(tf.nn.l2_loss(self.BU) 
 + tf.nn.l2_loss(self.BI) +  tf.nn.l2_loss(self.WPU) + tf.nn.l2_loss(self.WPI)) + 0.5 * self.config.lamb5 * tf.nn.l2_loss(self.bias_item_binvalue) + 0.5*self.config.lamb6*tf.nn.l2_loss(self.bias_user_tvalue) + 0.5 * self.config.lamb7 * tf.nn.l2_loss(self.BTDay) + 0.5 * self.config.lamb8 * (tf.nn.l2_loss(self.WCU) + tf.nn.l2_loss(self.BCU)) + 0.5 * self.config.lamb9 * (tf.nn.l2_loss(self.WPUKT) + tf.nn.l2_loss(self.AlphaUK))

        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from l2_loss.
        Returns:
            train_op: The Op for training.
        """

        train_op = v1.train.AdamOptimizer(learning_rate = self.config.lr).minimize(loss,var_list=None)

        return train_op

    def train_on_batch(self, sess, input_df, mean_rank, mean_u_day): #, dev_set):

        feed = self.create_feed_dict(input_df, mean_rank, mean_u_day)
        _, pred = sess.run([self.train_op, self.pred], feed_dict=feed)
        return pred

    def run_epoch(self, sess, train_df, mean_rank, mean_u_day, dev_df): 
        
        shuffled_df = shuffle(train_df)
        
        num_loop = (len(shuffled_df.index) + 1) // self.config.batch_size if (len(shuffled_df.index) + 1) % self.config.batch_size == 0 else (len(shuffled_df.index) + 1 ) // self.config.batch_size + 1
        #print(num_loop)
        for i in range(num_loop):
            self.train_on_batch(sess, shuffled_df[i*self.config.batch_size: (i+1)*self.config.batch_size], mean_rank, mean_u_day)
        
        #print ("Evaluating on Test set")
                                                     
        ###Evaluate on DEV SET                                                                     
        dev_pred = sess.run(self.test_pred, feed_dict=self.create_feed_dict(dev_df, mean_rank, mean_u_day))                      
        dev_loss = sum((dev_pred - dev_df["overall"])**2) / len(dev_pred)                    
        return dev_loss
    
    ### Run_epoch without using mini batch 
    # def run_epoch(self, sess, train_df, mean_rank, mean_u_day, dev_df): 

    #     self.train_on_batch(sess,train_df , mean_rank, mean_u_day)
    #     # feed = self.create_feed_dict(train_df, mean_rank, mean_u_day)
    #     # _, pred = sess.run([self.train_op, self.pred], feed_dict=feed)
    #     print ("Evaluating on Test set")
    #     ###Evaluate on DEV SET                                                                     
    #     dev_pred = sess.run(self.test_pred, feed_dict=self.create_feed_dict(dev_df, mean_rank, mean_u_day))                      
    #     dev_loss = sum((dev_pred - dev_df["overall"])**2) / len(dev_pred)                    
    #     return dev_loss
    

    def fit(self, sess, saver, train_df, dev_df):
        best_dev_UAS = 1000

        mean_rank = train_df["overall"].mean() #global rank mean
    
        mean_u_day = getMeanDaybyUser(train_df)
        dev = []

        for epoch in range(self.config.n_epochs):
            print ("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
            dev_UAS = self.run_epoch(sess, train_df, mean_rank, mean_u_day, dev_df)
            dev.append(dev_UAS)
            ### Uncomment to print DEV loss a single epoch

            print("current DEV loss = ", dev_UAS)

            if dev_UAS < best_dev_UAS:
                best_dev_UAS = dev_UAS
                print('new best dev loss!:', best_dev_UAS)
                if saver:
                    print( "New best dev UAS! " + self.config.weight_filename)
                    saver.save(sess, self.config.weight_filename)
            print()
        self.draw_chart(sess, self.config.n_epochs,dev)
            #self.suggest_top_n_highest_scores(sess, dev_df, mean_rank, mean_u_day)
    def draw_chart(self,sess, n_epochs, dev_UAS):
        epochs = [i for i in range(1, n_epochs+1)]
        plt.plot(epochs, dev_UAS)
        plt.xlabel("Number of Epochs")
        plt.ylabel("dev_UAS")
        plt.title("Collaborative Filtering with Temporal Dynamics")
        plt.show()
    # def suggest_top_n_highest_scores(self, sess, dev_df, mean_rank, mean_u_day):
    #     # Select a specific user to suggest Top-N highest scores for
    #     user = dev_df[dev_df['userID'] == 1041]

    #     # Predict scores for the selected user
    #     user_pred = sess.run(self.test_pred, feed_dict=self.create_feed_dict(user, mean_rank, mean_u_day))

    #     # Combine the prediction scores with the product id
    #     user_pred_df = pd.DataFrame({"itemID": user['itemID'], "score": user_pred})

    #     # Sort the scores in descending order and select the Top-N
    #     user_pred_df.sort_values("score", ascending=False, inplace=True)
    #     top_n = user_pred_df[:self.config.top_n]

    #     print("Top-N highest scores for user {}:".format(1041))
    #     print(top_n)

    # def predict_user_item_score(self, sess, dev_df, mean_rank, mean_u_day, specific_user_id, specific_product_id):
    #     # # Select data for the specific user and item
    #     # user_item = dev_df[(dev_df['userID'] == specific_user_id) & (dev_df['itemID'] == specific_product_id)]

    #     # # Predict score for the specific user and item
    #     # score = sess.run(self.test_pred, feed_dict=self.create_feed_dict(user_item, mean_rank, mean_u_day))

    #     # print("Score for user {} on item {}:".format(specific_user_id, specific_product_id))
    #     # print(score)

    #     # Predict scores for all users
    #     all_user_pred = sess.run(self.test_pred, feed_dict=self.create_feed_dict(dev_df, mean_rank, mean_u_day))

    #     # Combine the prediction scores with the product id and user id
    #     all_user_pred_df = pd.DataFrame({"itemID": dev_df['itemID'], "userID": dev_df['userID'], "score": all_user_pred})

    #     # Pivot the data to get the matrix of user-item scores
    #     matrix_user_item = all_user_pred_df.pivot(index='userID', columns='itemID', values='score')

    #     print("Matrix of user-item scores:")
    #     print(matrix_user_item)


    def __init__(self, config): 
        self.config = config
        self.build()

