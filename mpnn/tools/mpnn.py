from keras import backend as K
from .tools import stretch_array
import tensorflow as tf
import numpy as np

def build_network(X_nodes, X_edges, X_nodes_in_out, 
                  X_messages_in, X_messages_out, message_passers, 
                  state_updater, readout, ndim_features_nodes, fake_message_const, steps):
    for step in range(steps):
        messages = message_passers[step](
            K.concatenate(
                [
                    K.reshape(K.gather(reference=X_nodes, indices=X_nodes_in_out), shape=(-1, 2 * ndim_features_nodes)), 
                    X_edges
                ], axis=1
            )
        )
        messages = K.concatenate([messages, fake_message_const], axis=0)
        messages = tf.where(tf.is_inf(messages), tf.zeros_like(messages), messages)

        messages_aggregated_in = K.max(K.gather(reference=messages, indices=X_messages_in), axis=1)
        messages_aggregated_out = K.max(K.gather(reference=messages, indices=X_messages_out), axis=1)
        
        messages_aggregated_in2 = K.mean(K.gather(reference=messages, indices=X_messages_in), axis=1)
        messages_aggregated_out2 = K.mean(K.gather(reference=messages, indices=X_messages_out), axis=1)
        
        messages_aggregated_in3 = K.var(K.gather(reference=messages, indices=X_messages_in), axis=1)
        messages_aggregated_out3 = K.var(K.gather(reference=messages, indices=X_messages_out), axis=1)
        
        messages_aggregated_in4 = K.std(K.gather(reference=messages, indices=X_messages_in), axis=1)
        messages_aggregated_out4 = K.std(K.gather(reference=messages, indices=X_messages_out), axis=1)
       
        
        ## For GRU-based state_updater
      #  _, X_nodes = state_updater(
      #     inputs=K.concatenate([messages_aggregated_in, messages_aggregated_out
                              #      ,messages_aggregated_in2, messages_aggregated_out2, 
                               #      messages_aggregated_in3, messages_aggregated_out3,
                                #    ], axis=1),
           # state=X_nodes
     #   )
        
       # For LSTM-based state_updater
      #  _, (_, X_nodes) = state_updater(
       #     inputs=K.concatenate([messages_aggregated_in, messages_aggregated_out
        #                          ,messages_aggregated_in2, messages_aggregated_out2, 
         #                         messages_aggregated_in3, messages_aggregated_out3, 
          #                        messages_aggregated_in4, messages_aggregated_out4
             #                    ], axis=1),
         #   state=(tf.zeros_like(X_nodes), X_nodes)
     #   )
        
        ## For dense state_updater
        X_nodes = state_updater(K.concatenate([messages_aggregated_in, messages_aggregated_in2, messages_aggregated_out, messages_aggregated_out2, messages_aggregated_in3, messages_aggregated_out3, 
                                               messages_aggregated_in4, 
                                              messages_aggregated_out4, 
                                             X_nodes], axis=1))
        
    return readout(X_nodes)

    
def run_train(X_cluster_graph, X_predictions, optimizer, sess, 
              ndim_features_nodes, ndim_features_edges, placeholders, metrics=[]):
    _, predictions, *metrics = sess.run([optimizer, X_predictions] + metrics, feed_dict={
        placeholders['X_nodes']: stretch_array(X_cluster_graph['X_cluster_nodes'], ndim_features_nodes),
        placeholders['X_edges']: stretch_array(X_cluster_graph['X_cluster_edges'], ndim_features_edges),
        placeholders['X_labels']: X_cluster_graph['Y_cluster_labels'],
        placeholders['X_nodes_in_out']: X_cluster_graph['X_cluster_in_out'],
        placeholders['X_messages_in']: X_cluster_graph['X_cluster_messages_in'], 
        placeholders['X_messages_out']: X_cluster_graph['X_cluster_messages_out'],
        K.learning_phase(): 1
    })
    
    return predictions, metrics


def run_test(X_cluster_graph, X_predictions, sess,
             ndim_features_nodes, ndim_features_edges, placeholders):
    predictions = sess.run(X_predictions, feed_dict={
        placeholders['X_nodes']: stretch_array(X_cluster_graph['X_cluster_nodes'], ndim_features_nodes),
        placeholders['X_edges']: stretch_array(X_cluster_graph['X_cluster_edges'], ndim_features_edges),
        placeholders['X_nodes_in_out']: X_cluster_graph['X_cluster_in_out'],
        placeholders['X_messages_in']: X_cluster_graph['X_cluster_messages_in'],
        placeholders['X_messages_out']: X_cluster_graph['X_cluster_messages_out'],
        K.learning_phase(): 0
    })

    return predictions
