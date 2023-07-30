import os
import time
import tensorflow as tf
import tensorflow.compat.v1 as v1
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import Model
from w_initialization import xavier_weight_init
from utils.general_utils import Progbar
from DataProcessing import getMeanDay, getMeanDaybyUser, getUserRatedItemCount

from sklearn.utils import shuffle


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    # n_items = 36
    # n_users = 38
    #n_mean = 0
    n_f = 5
    # lamb2 = 25
    # lamb3 = 10
    lamb4 = 0.02 #0.02
    lamb5 = 0.01 #10
    lamb6 = 0.5 #15
    lamb7 = 0.1
    lamb8 = 0.1
    lamb9 = 10
    lamb10 = 0.1
    beta = 0.4
    n_epochs = 30
    lr = 0.001 #0.005
    item_bin_size = 60
    batch_size = 32
    weight_filename = "data\\weights\\recommanding_with_IU_time_drifting.weight"
    os.makedirs(weight_filename, exist_ok=True)
    device="/cpu:0"
    maxday_cat_code = 32
    
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
        
        self.user_itemcount_placeholder = v1.placeholder(tf.int32, shape = [None, ], name = "user_rated_item_count")
        self.user_rated_item_placeholder = v1.placeholder(tf.int32, shape = [None, None], name = "y_placeholder")


    def create_feed_dict(self, input_df, mean_rank, mean_u_day, global_ru_df): #label_indices, label_data, label_shape, input_rank_data, input_tb_data, input_td_data, mean_rank, mean_u_day):
        feed_dict = {}

        feed_dict[self.user_placeholder] = input_df["userID"].values.astype(int)
        feed_dict[self.item_placeholder] = input_df["itemID"].values.astype(int)
        feed_dict[self.rank_placeholder] = input_df["overall"].values
        feed_dict[self.tbin_placeholder] = input_df["ITBin"].values.astype(int)
        feed_dict[self.tday_placeholder] = input_df["ReviewDay"].values.astype(int)
        feed_dict[self.global_mean_placeholder] = mean_rank
        feed_dict[self.mean_ud_placeholder] = mean_u_day.astype(float)
        feed_dict[self.maxday_cat_placeholder] = input_df["TDayCat"].values.astype(int) #max_c_day     
        
        feed_dict[self.user_itemcount_placeholder] = global_ru_df["itemLen"].values.astype(int)  #getUserRatedItemCount(input_df)
        feed_dict[self.user_rated_item_placeholder] = global_ru_df["itemList"].tolist() #.astype(int)
        
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
            
            #implicit item vectors
            self.Y = v1.get_variable("Y", shape = [self.config.n_items, self.config.n_f], 
                                   initializer =  weight_initializer)
            
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
            
            #add an extra row of all zeros to fit for the patched values
            y_w_extra = tf.zeros([1, self.config.n_f])
            y_w = tf.concat([self.Y, y_w_extra], 0)
            
            y_js = tf.nn.embedding_lookup(y_w, self.user_rated_item_placeholder, name = "y_j_sum")
         
        with tf.device(self.config.device):             
            tday_diff = tf.subtract(tf.cast(self.tday_placeholder,tf.float32), mean_tday) # t - t(mean)            
            dev_t = tf.multiply(tf.sign(tday_diff), tf.pow(tf.abs(tday_diff), self.config.beta)) #sign(t - t(mean))*abolute(t - t(mean))**beta       
            self.bias_user_tvalue =  tf.multiply(alpha_value, dev_t)
            bias_user_time = tf.add(bias_user, self.bias_user_tvalue)
            bias_user_time = tf.add(bias_user_time, butday)
            bias_item_time = tf.add(bias_item, self.bias_item_binvalue)
            cui = tf.add(cu_b, cu_t)
            bias_item_time = tf.multiply(bias_item_time, cui) #cong thuc 12 trong paper
            
            self.vector_user_tvalue = tf.transpose(tf.multiply(tf.transpose(alpha_uk_value), dev_t))
            
            y_sum = tf.reduce_sum(y_js, axis = 1)
            y_sum_el = tf.nn.embedding_lookup(y_sum, self.user_placeholder)
            #compute R(U)**(-0.5)
            
            ru_by_user = tf.pow(tf.cast(self.user_itemcount_placeholder,tf.float32), -0.5)
            ru_list = tf.nn.embedding_lookup(ru_by_user, self.user_placeholder)
            y_implicit = tf.transpose(tf.multiply(tf.transpose(y_sum_el), ru_list, name="y_sum_temp"), name = "y_implicit_sum")
            user_vector_implicit = tf.add(user_vector, y_implicit)
            
            user_vector_t = tf.add(user_vector_implicit, self.vector_user_tvalue)
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
 + tf.nn.l2_loss(self.BI) +  tf.nn.l2_loss(self.WPU) + tf.nn.l2_loss(self.WPI)) + 0.5 * self.config.lamb5 * tf.nn.l2_loss(self.bias_item_binvalue) + 0.5*self.config.lamb6*tf.nn.l2_loss(self.bias_user_tvalue) + 0.5 * self.config.lamb7 * tf.nn.l2_loss(self.BTDay) + 0.5 * self.config.lamb8 * (tf.nn.l2_loss(self.WCU) + tf.nn.l2_loss(self.BCU)) + 0.5 * self.config.lamb9 * (tf.nn.l2_loss(self.WPUKT) + tf.nn.l2_loss(self.AlphaUK)) + 0.5*self.config.lamb10*(tf.nn.l2_loss(self.Y))

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

        train_op = v1.train.AdamOptimizer(learning_rate = self.config.lr).minimize(loss)

        return train_op

    def train_on_batch(self, sess, input_df, mean_rank, mean_u_day, global_ru_df): #, dev_set):

        feed = self.create_feed_dict(input_df, mean_rank, mean_u_day, global_ru_df)
        _, pred = sess.run([self.train_op, self.pred], feed_dict=feed)
        return pred

    def run_epoch(self, sess, train_df, mean_rank, mean_u_day, dev_df, global_ru_df): 
        
        shuffled_df = shuffle(train_df)
        
        num_loop = (len(shuffled_df.index) + 1) // self.config.batch_size if (len(shuffled_df.index) + 1) % self.config.batch_size == 0 else (len(shuffled_df.index) + 1 ) // self.config.batch_size + 1
        #print(num_loop)
        for i in range(num_loop):
            self.train_on_batch(sess, shuffled_df[i*self.config.batch_size: (i+1)*self.config.batch_size], mean_rank, mean_u_day, global_ru_df)
        
        print ("Evaluating on dev set")
                                                     
        ###Evaluate on DEV SET                                                                     
        dev_pred = sess.run(self.test_pred, feed_dict=self.create_feed_dict(dev_df, mean_rank, mean_u_day, global_ru_df))                      
        dev_loss = sum((dev_pred - dev_df["overall"])**2) / len(dev_pred)                    
        return dev_loss

    def fit(self, sess, saver, train_df, dev_df, global_ru_df):
        best_dev_UAS = 1000

        mean_rank = train_df["overall"].mean() #global rank mean
        
        mean_u_day = getMeanDaybyUser(train_df)
        dev = []

        for epoch in range(self.config.n_epochs):
            print ("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
            dev_UAS = self.run_epoch(sess, train_df, mean_rank, mean_u_day, dev_df, global_ru_df)
            dev.append(dev_UAS)
            print("current DEV loss = ", dev_UAS)

            if dev_UAS < best_dev_UAS:
                best_dev_UAS = dev_UAS
                print('new best dev loss!:', best_dev_UAS)
                if saver:
                    print( "New best dev UAS! " + self.config.weight_filename)
                    saver.save(sess, self.config.weight_filename)
            print()
        self.draw_chart(sess, self.config.n_epochs,dev)

    def draw_chart(self,sess, n_epochs, dev_UAS):
        epochs = [i for i in range(1, n_epochs+1)]
        plt.plot(epochs, dev_UAS)
        plt.xlabel("Number of Epochs")
        plt.ylabel("dev_UAS")
        plt.title("Collaborative Filtering with Temporal Dynamics")
        plt.show()

    def calculate_precision_at_k(self, sess,train_df, test_df,mean_rank, mean_u_day,global_ru_df, k):
        user_ids = test_df['userID'].unique()
        num_users = len(user_ids)
        precision_sum = 0
        
        mean_rank = train_df["overall"].mean() #global rank mean
        
        mean_u_day = getMeanDaybyUser(train_df)
        for i, user_id in enumerate(user_ids):
            if i % 100 == 0:
                print("Calculating precision for user {:} out of {:}".format(i+1, num_users))
        
            user_ratings = test_df[test_df['userID'] == user_id] # lấy ra list các item được rate bởi user i
            if len(user_ratings) < k:
                continue
        
            item_scores = sess.run(self.test_pred, feed_dict=self.create_feed_dict(user_ratings, mean_rank, mean_u_day, global_ru_df))
            item_scores = pd.Series(item_scores, index=user_ratings.index)
            item_scores = item_scores.sort_values(ascending=False)[:k]
            print(item_scores)
            relevant_items = user_ratings[user_ratings['overall'] >= 1.0]['itemID']  # lấy các rating >3 trong tập user_ratings
            #print(relevant_items)
            num_relevant_items = len(relevant_items)

            if num_relevant_items == 0:
                continue
        
            num_retrieved_items = len(item_scores)
            num_retrieved_relevant_items = len(relevant_items[relevant_items.isin(item_scores.index)])
            #print(relevant_items[relevant_items.isin(item_scores.index)])
            precision_sum += num_retrieved_relevant_items / num_retrieved_items
            
    
        return precision_sum 

    def __init__(self, config): 
        self.config = config
        self.build()

