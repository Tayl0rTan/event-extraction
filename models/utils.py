import tensorflow as tf
import tensorflow_probability as tfp
import torch.nn as nn
nn.KLDivLoss()

def cal_binary_dsc_loss(logits,labels,seq_mask,num_labels,one_hot=True,smoothing_lambda=1.0):
    # 这里暂时不用mask，因为mask的地方，label都是0，会被忽略掉
    if one_hot:
        labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    else:
        labels = tf.expand_dims(labels,axis=-1)
    # seq_mask = tf.cast(seq_mask, tf.float32)
    predict_prob = tf.nn.softmax(logits, axis=-1, name="predict_prob")
    pos_prob = predict_prob[:, :, 1]
    neg_prob = predict_prob[:, :, 0]
    pos_label = labels[:, :, 1]
    nominator = neg_prob * pos_prob * pos_label
    denominator = neg_prob * pos_prob + pos_label
    loss = (nominator + smoothing_lambda)/(denominator + smoothing_lambda)
    loss = 1. - loss
    loss = tf.reduce_sum(loss,axis=-1)
    loss = tf.reduce_mean(loss)
    return loss

def dice_dsc_loss(logits,labels,text_length_list,seq_mask,slot_label_num,smoothing_lambda=1.0):
    """
    dice loss dsc
    :param logits: [batch_size,time_step,num_class]
    :param labels: [batch_size,time_step]
    :param seq_length:[batch_size]
    :return:
    """

    predict_prob = tf.nn.softmax(logits,axis=-1,name="predict_prob")
    label_one_hot = tf.one_hot(labels, depth=slot_label_num, axis=-1)
    # seq_mask = tf.sequence_mask(seq_mask)
    # seq_mask = tf.cast(seq_mask,dtype=tf.float32)
    # batch_size_tensor = tf.range(0,tf.shape(logits)[0])
    # seq_max_len_tensor = tf.range(0,tf.shape(logits)[1])
    # # batch_size_tensor = tf.expand_dims(batch_size_tensor,1)
    # seq_max_len_tensor = tf.expand_dims(seq_max_len_tensor,axis=0)
    # seq_max_len_tensor = tf.tile(seq_max_len_tensor,[tf.shape(logits)[0],1])
    # seq_max_len_tensor = tf.expand_dims(seq_max_len_tensor,axis=-1)
    # batch_size_tensor = tf.expand_dims(batch_size_tensor, 1)
    # batch_size_tensor = tf.tile(batch_size_tensor, [1, tf.shape(logits)[1]])
    # batch_size_tensor = tf.expand_dims(batch_size_tensor, -1)
    # # batch_zeros_result = tf.zeros((tf.shape(logits)[0],tf.shape(logits)[1],3), dtype=tf.int32)
    # labels = tf.expand_dims(labels,axis=-1)
    # gather_idx = tf.concat([batch_size_tensor,seq_max_len_tensor,labels],axis=-1)
    # gather_result = tf.gather_nd(predict_prob,gather_idx)
    # # gather_result = gather_result
    # neg_prob = 1. - gather_result
    # neg_prob = neg_prob
    # # gather_result = gather_result
    # cost = 1. - neg_prob*gather_result/(neg_prob*gather_result+1.)
    # cost = cost * seq_mask
    # cost = tf.reduce_sum(cost,axis=-1)
    # cost = tf.reduce_mean(cost)
    # return cost
    # neg_prob = 1.- predict_prob
    nominator = 2*predict_prob*label_one_hot+smoothing_lambda
    denomiator = predict_prob*predict_prob+label_one_hot*label_one_hot+smoothing_lambda
    result = nominator/denomiator
    result = 1. - result
    result = tf.reduce_sum(result,axis=-1)
    result = result * seq_mask
    result = tf.reduce_sum(result,axis=-1,keep_dims=True)
    result = result/tf.cast(text_length_list,tf.float32)
    result = tf.reduce_mean(result)
    return result
    # cost = cal_binary_dsc_loss(predict_prob[:, :, 0],label_one_hot[:, :, 0],seq_mask)
    # for i in range(1,slot_label_num):
    #     cost += cal_binary_dsc_loss(predict_prob[:, :, i],label_one_hot[:, :, i],seq_mask)
    #     # print(denominator)
    # cost = cost/float(slot_label_num)
    # cost = tf.reduce_mean(cost)
    # return cost

def vanilla_dsc_loss(logits,labels,seq_mask,num_labels,smoothing_lambda=1.0,one_hot=True):
    if one_hot:
        labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    else:
        labels = tf.expand_dims(labels,axis=-1)

    predict_prob = tf.nn.softmax(logits, axis=-1, name="predict_prob")
    pos_prob = predict_prob[:, :, 1]
    neg_prob = predict_prob[:,:,0]
    pos_label = labels[:, :, 1]
    neg_label = labels[:,:,0]
    denominator = 2 * pos_prob * pos_label + neg_prob * pos_label + pos_prob * neg_label + smoothing_lambda
    nominator = 2 * pos_prob * pos_label + smoothing_lambda
    loss = 1. - nominator / denominator
    loss = loss * tf.cast(seq_mask,tf.float32)
    loss = tf.reduce_sum(loss,axis=-1)
    loss = tf.reduce_mean(loss,axis=0)
    return loss

def dl_dsc_loss(logits,labels,text_length_list,seq_mask,slot_label_num,smoothing_lambda=1.0,gamma=2.0):
    predict_prob = tf.nn.softmax(logits, axis=-1, name="predict_prob")
    label_one_hot = tf.one_hot(labels, depth=slot_label_num, axis=-1)
    # neg_prob = 1.- predict_prob
    # neg_prob = tf.pow(neg_prob,gamma)
    pos_prob = predict_prob[:,:,1]
    pos_prob_squre = tf.pow(pos_prob,2)
    pos_label = label_one_hot[:,:,1]
    pos_label_squre = tf.pow(pos_label,2)
    nominator = 2*pos_prob_squre*pos_label_squre+smoothing_lambda
    denominator = pos_label_squre+pos_label_squre+smoothing_lambda
    result = nominator/denominator
    result = 1.-result
    result = result * tf.cast(seq_mask,tf.float32)
    result = tf.reduce_sum(result, axis=-1, keep_dims=True)
    result = result / tf.cast(text_length_list, tf.float32)
    result = tf.reduce_mean(result)
    return result

def ce_loss(logits,labels,mask,num_labels,one_hot=True,imbalanced_ratio=2):
    if one_hot:
        labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    else:
        labels = tf.expand_dims(labels,axis=-1)
    probs = tf.nn.softmax(logits,axis=-1)
    pos_probs = probs[:,:,1]
    pos_probs = tf.pow(pos_probs,imbalanced_ratio)
    pos_probs = tf.expand_dims(pos_probs, axis=-1)
    neg_probs = 1. - pos_probs
    probs = tf.concat([neg_probs,pos_probs],axis=-1)
    print(probs)
    log_probs = tf.log(probs+1e-7)
    per_example_loss = -tf.reduce_sum(tf.cast(labels,tf.float32) * log_probs, axis=-1)
    per_example_loss = per_example_loss * tf.cast(mask, tf.float32)
    loss = tf.reduce_sum(per_example_loss, axis=-1)
    loss = tf.reduce_mean(loss)
    return loss

def focal_loss(logits,labels,mask,num_labels,one_hot=True,lambda_param=1.5):
    probs = tf.nn.softmax(logits,axis=-1)
    pos_probs = probs[:,:,1]
    prob_label_pos = tf.where(tf.equal(labels,1),pos_probs,tf.ones_like(pos_probs))
    prob_label_neg = tf.where(tf.equal(labels,0),pos_probs,tf.zeros_like(pos_probs))
    loss = tf.pow(1. - prob_label_pos,lambda_param)*tf.log(prob_label_pos + 1e-7) + \
           tf.pow(prob_label_neg,lambda_param)*tf.log(1. - prob_label_neg + 1e-7)
    loss = -loss * tf.cast(mask,tf.float32)
    loss = tf.reduce_sum(loss,axis=-1,keepdims=True)
    # loss = loss/tf.cast(tf.reduce_sum(mask,axis=-1),tf.float32)
    loss = tf.reduce_mean(loss)
    return loss

def span_loss(logits,labels,mask):
    probs = tf.nn.softmax(logits,axis=1)
    arg_max_label = tf.cast(tf.where(probs > 0.5,tf.ones_like(labels),tf.zeros_like(labels)),tf.int32)
    arg_max_label *= mask

    
def test2():
    const_var = tf.constant([[0,0,0,1,0,1], [0,0, 0, 1,0,0], [0,0,1,0,0,0], [0, 1, 0,1,1,0]])
    a = tf.expand_dims(const_var,axis=-1)
    b = tf.where(a)
    return b

if __name__ == "__main__":
    result = test2()


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(result))
